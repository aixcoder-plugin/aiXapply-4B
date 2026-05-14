import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from aiolimiter import AsyncLimiter
from datasets import load_dataset
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError
from tqdm import tqdm

from prompt_opencode_sar import PROMPT_OPENCODE_SAR
from search_and_replace_tool import SearchAndReplaceError, apply_search_and_replace

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    # Optional dependency: keep runnable even if python-dotenv isn't installed in the runtime.
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SCRIPT_DIR = Path(__file__).resolve().parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch run search_and_replace: call an OpenAI-compatible endpoint to generate JSON edits "
            "(OpenCode edit-tool style), apply them to original_code, and write results to jsonl."
        )
    )
    parser.add_argument(
        "-d",
        "--data-path",
        required=True,
        help="Path to the evaluation parquet file (or a datasets-compatible path).",
    )
    parser.add_argument(
        "--filter-num",
        type=int,
        default=0,
        help="Filter the dataset by the number of examples.",
    )
    parser.add_argument(
        "-p",
        "--provider",
        default="glm-5",
        choices=["glm-5", "qwen3.5-397b", "deepseek-v3.2", "kimi-k2.5", "minimax-m2.5", "aiXapply", "custom"],
        help="Model name.",
    )
    parser.add_argument("--api-key", default=None, help="API key override (OpenAI-compatible).")
    parser.add_argument("--base-url", default=None, help="Base URL override (OpenAI-compatible).")
    parser.add_argument("--model", default=None, help="Model name override.")
    parser.add_argument(
        "--extra-body",
        default=None,
        help='Optional JSON string for request extra_body (e.g. \'{"temperature":0.3}\').',
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32 * 1024,
        help="Maximum tokens to generate for each request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p nucleus sampling value.",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=16,
        help="Maximum concurrent requests to send.",
    )
    parser.add_argument(
        "-r",
        "--rate-limit",
        type=int,
        default=120,
        help="Maximum requests allowed in the rate window.",
    )
    parser.add_argument(
        "--rate-period",
        type=int,
        default=60,
        help="Rate limit window size in seconds.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1000.0,
        help="Timeout (seconds) for each request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry attempts for recoverable errors.",
    )
    parser.add_argument(
        "--fast-sar",
        action="store_true",
        default=False,
        help=(
            "Use fast SAR mode: build SAR prompt from columns {original_code, update_snippet, language}. "
            "Otherwise, extract <source_file>/<update_snippet> from example['prompt'] like infer_openai.py."
        ),
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output jsonl path. Default: predictions/{model}_sar_{timestamp}.jsonl",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Existing jsonl path to resume from. Records whose index/idx already exists "
            "with non-empty prediction_patch will be skipped. If --output-path is omitted, resume in place."
        ),
    )
    return parser.parse_args()


def extract_user_content(messages: Iterable[Dict[str, Any]]) -> str:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def extract_tagged_blocks(user_prompt: str) -> Tuple[str, str]:
    def _extract(tag: str) -> str:
        start = f"<{tag}>"
        end = f"</{tag}>"
        if start in user_prompt and end in user_prompt:
            return user_prompt.split(start, 1)[1].split(end, 1)[0]
        return ""

    return _extract("source_file"), _extract("update_snippet")


def build_sar_messages(language: str, source_file: str, update_snippet: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": PROMPT_OPENCODE_SAR["system"]},
        {
            "role": "user",
            "content": PROMPT_OPENCODE_SAR["user"].format(
                language=language,
                source_file=source_file,
                update_snippet=update_snippet,
            ),
        },
    ]

sampling_one = True
async def generate_completion(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    messages: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Optional[str], float]:
    attempt = 0
    backoff_factor = 2
    global sampling_one

    while attempt <= args.max_retries:
        try:
            async with limiter:
                start_time = time.time()
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=args.model,
                        stream=False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        messages=messages,
                        extra_body=args.extra_body or {},
                    ),
                    timeout=args.request_timeout,
                )
                end_time = time.time()
            if sampling_one:
                logger.info(f"Sampling one response: \n{response}")
                sampling_one = False
            return response.choices[0].message.content, end_time - start_time
        except asyncio.TimeoutError:
            logger.error("Request timed out after %.1f seconds", args.request_timeout)
        except APITimeoutError:
            logger.error("OpenAI API request timed out")
        except APIConnectionError as ce:
            logger.error("Connection error (server disconnected / network issue): %s", ce)
        except APIStatusError as se:
            status = se.status_code
            logger.error("HTTP error %s: %s", status, se.message)
            if status not in {429, 500, 502, 503}:
                return None, 0.0
        except Exception:
            logger.exception("Unexpected error while requesting completion")

        attempt += 1
        if attempt <= args.max_retries:
            wait_time = backoff_factor**attempt + 10
            logger.info("Retrying in %s seconds (attempt %s/%s)", wait_time, attempt, args.max_retries)
            await asyncio.sleep(wait_time)
        else:
            return None, 0.0
    return None, 0.0


def _get_language(example: Dict[str, Any]) -> str:
    # Prefer dataset language columns if present; fallback to extra_info
    lang = example.get("language")
    if isinstance(lang, str) and lang.strip():
        return lang
    extra = example.get("extra_info") or {}
    if isinstance(extra, dict):
        lang2 = extra.get("language")
        if isinstance(lang2, str) and lang2.strip():
            return lang2
    return ""


def _get_ground_truth(example: Dict[str, Any]) -> Optional[str]:
    # Prefer final_code if present, else reward_model.ground_truth
    gt = example.get("final_code")
    if isinstance(gt, str):
        return gt
    reward_model = example.get("reward_model") or {}
    if isinstance(reward_model, dict):
        gt2 = reward_model.get("ground_truth")
        if isinstance(gt2, str):
            return gt2
    return None


def build_record_sar(
    idx: int,
    language: str,
    original_code: str,
    update_snippet: str,
    prediction_patch: Optional[str],
    elapsed_time: float,
    applied_code: Optional[str],
    apply_error: Optional[str],
    ground_truth: Optional[str],
) -> Dict[str, Any]:
    return {
        "index": int(idx),
        "language": language,
        "prediction_patch": prediction_patch,
        "elapsed_time": elapsed_time,
        "prediction": applied_code,
        "apply_error": apply_error,
        "ground_truth": ground_truth,
        "original_code": original_code,
        "update_snippet": update_snippet,
    }

def _add_line_numbers(text: str, *, prefix: str = "L") -> str:
    # Use \n as display newline; this is a reference view for the model only.
    lines = text.splitlines()
    return "\n".join(f"{prefix}{i + 1}:{line}" for i, line in enumerate(lines))

async def process_example_sar(
    idx: int,
    example: Dict[str, Any],
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Any]]:
    language = _get_language(example)

    if args.fast_sar:
        original_code = str(example.get("original_code") or "")
        update_snippet = str(example.get("update_snippet") or "")
        ground_truth = _get_ground_truth(example)
    else:
        messages = example.get("prompt") or []
        if not isinstance(messages, list):
            raise ValueError(f"Unexpected prompt format at index {idx}: {messages!r}")
        user_prompt = extract_user_content(messages)
        original_code, update_snippet = extract_tagged_blocks(user_prompt)
        ground_truth = _get_ground_truth(example)

    original_code_with_line_numbers = _add_line_numbers(original_code)
    sar_messages = build_sar_messages(language=language, source_file=original_code_with_line_numbers, update_snippet=update_snippet)
    prediction_patch, elapsed_time = await generate_completion(client, limiter, sar_messages, args)

    applied_code: Optional[str] = None
    apply_error: Optional[str] = None
    if prediction_patch:
        try:
            applied_code = apply_search_and_replace(original_code, prediction_patch)
        except SearchAndReplaceError as e:
            apply_error = str(e)
        except Exception as e:
            apply_error = f"Unexpected apply error: {e}"
    else:
        apply_error = "Empty model output"
    
    if applied_code is None:
        applied_code = original_code

    record = build_record_sar(
        idx=idx,
        language=language,
        original_code=original_code,
        update_snippet=update_snippet,
        prediction_patch=prediction_patch,
        elapsed_time=elapsed_time,
        applied_code=applied_code,
        apply_error=apply_error,
        ground_truth=ground_truth,
    )
    return idx, record


def _load_dataset_auto(data_path: Path, fast_sar: bool):
    """
    A slightly more robust loader than infer_openai.py:
    - If data_path is a parquet file -> load_dataset('parquet', data_files=...)
    - Else -> load_dataset(data_path) (expects datasets-compatible source)
    """
    if data_path.is_file() and data_path.suffix.lower() == ".parquet":
        return load_dataset("parquet", data_files={"data": str(data_path)})["data"]
    if fast_sar:
        return load_dataset(str(data_path))["train"]
    return load_dataset(str(data_path))["train"]


def _parse_extra_body(extra_body: Optional[str]) -> Dict[str, Any]:
    if not extra_body:
        return {"enable_thinking": False, "thinking": {"type": "disabled"}}
    try:
        obj = json.loads(extra_body)
    except Exception as e:
        raise ValueError(f"--extra-body must be valid JSON. got error: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("--extra-body must be a JSON object.")
    return {"enable_thinking": False, "thinking": {"type": "disabled"}, **obj}


def _extract_record_index(record: Dict[str, Any]) -> Optional[int]:
    raw_index = record.get("index")
    if raw_index is None:
        raw_index = record.get("idx")
    if raw_index is None:
        return None
    try:
        return int(raw_index)
    except (TypeError, ValueError):
        return None


def _record_has_prediction_output(record: Dict[str, Any]) -> bool:
    prediction_patch = record.get("prediction_patch")
    return isinstance(prediction_patch, str) and bool(prediction_patch.strip())


def _load_resume_state(resume_path: Path, dataset_size: int) -> Tuple[set[int], List[str]]:
    if not resume_path.exists():
        raise FileNotFoundError(f"Could not find resume file at {resume_path}")

    completed_indices: set[int] = set()
    preserved_lines: List[str] = []
    invalid_count = 0
    duplicate_count = 0
    out_of_range_count = 0
    retryable_count = 0

    with resume_path.open("r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_count += 1
                continue
            if not isinstance(record, dict):
                invalid_count += 1
                continue

            idx = _extract_record_index(record)
            if idx is None:
                invalid_count += 1
                continue
            if idx < 0 or idx >= dataset_size:
                out_of_range_count += 1
                continue
            if not _record_has_prediction_output(record):
                retryable_count += 1
                continue
            if idx in completed_indices:
                duplicate_count += 1
                continue

            completed_indices.add(idx)
            preserved_lines.append(line)

    logger.info(
        "Loaded %s completed records from %s",
        len(completed_indices),
        resume_path,
    )
    if duplicate_count:
        logger.warning("Ignored %s duplicate indices while loading %s", duplicate_count, resume_path)
    if invalid_count:
        logger.warning("Ignored %s invalid resume lines while loading %s", invalid_count, resume_path)
    if out_of_range_count:
        logger.warning("Ignored %s out-of-range indices while loading %s", out_of_range_count, resume_path)
    if retryable_count:
        logger.info(
            "Will retry %s existing records with empty prediction output from %s",
            retryable_count,
            resume_path,
        )

    return completed_indices, preserved_lines


def _path_needs_leading_newline(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    with path.open("rb") as infile:
        infile.seek(-1, os.SEEK_END)
        return infile.read(1) != b"\n"


async def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")

    dataset = _load_dataset_auto(data_path, args.fast_sar)

    if args.filter_num > 0:
        dataset = dataset.select(range(args.filter_num))
        print("================================================")
        print("dataset[0]:", dataset[0])
        print("================================================")

    resume_path = Path(args.resume_from) if args.resume_from else None
    if args.output_path:
        output_path = Path(args.output_path)
    elif resume_path is not None:
        output_path = resume_path
    else:
        output_path = SCRIPT_DIR / "predictions" / f"{args.model}_search_and_replace_{run_timestamp}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("output_path:", str(output_path), flush=True)

    completed_indices: set[int] = set()
    preserved_lines: List[str] = []
    output_mode = "w"
    needs_leading_newline = False

    if resume_path is not None:
        completed_indices, preserved_lines = _load_resume_state(resume_path, len(dataset))
        logger.info(
            "Resume mode: %s/%s examples already completed, %s remaining",
            len(completed_indices),
            len(dataset),
            len(dataset) - len(completed_indices),
        )

        with output_path.open("w", encoding="utf-8") as out_file:
            for line in preserved_lines:
                out_file.write(line + "\n")
        output_mode = "a"
        needs_leading_newline = _path_needs_leading_newline(output_path)

        if len(completed_indices) >= len(dataset):
            logger.info("All %s examples are already present in %s", len(dataset), resume_path)
            return

    limiter = AsyncLimiter(args.rate_limit, args.rate_period)
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    progress = tqdm(total=len(dataset), desc="Requesting(SAR)", initial=len(completed_indices))
    buffer: Dict[int, Dict[str, Any]] = {}
    next_to_write = 0
    pending: set[asyncio.Task] = set()
    index_iter = iter(range(len(dataset)))

    skipped_indices: List[int] = []
    print("skipped_indices:", skipped_indices, flush=True)

    def advance_next_to_write() -> None:
        nonlocal next_to_write
        while next_to_write in completed_indices or next_to_write in skipped_indices:
            next_to_write += 1

    def enqueue() -> bool:
        index = -1
        while index == -1 or index in skipped_indices or index in completed_indices:
            try:
                index = next(index_iter)
            except StopIteration:
                return False

        if args.fast_sar:
            task = asyncio.create_task(process_example_sar(index, dataset[index], client, limiter, args))
        else:
            task = asyncio.create_task(process_example_sar(index, dataset[index], client, limiter, args))
        pending.add(task)
        return True

    for _ in range(min(args.concurrency, len(dataset))):
        enqueue()

    with output_path.open(output_mode, encoding="utf-8") as out_file:
        if needs_leading_newline:
            out_file.write("\n")
            out_file.flush()
        advance_next_to_write()
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    idx, record = task.result()
                except Exception:
                    logger.exception("Task failed")
                    progress.update(1)
                    continue

                buffer[idx] = record
                while next_to_write in buffer or next_to_write in skipped_indices:
                    if next_to_write in skipped_indices:
                        next_to_write += 1
                        continue
                    current = buffer.pop(next_to_write)
                    out_file.write(json.dumps(current, ensure_ascii=False) + "\n")
                    out_file.flush()
                    next_to_write += 1
                    advance_next_to_write()
                progress.update(1)

            while len(pending) < args.concurrency and enqueue():
                continue
    progress.close()
    logger.info("SAR predictions written to %s", output_path)


def main() -> None:
    args = parse_args()

    model_configs = {
        "glm-5": {
            "api_key": os.getenv("OPENAI_KEY", ""),
            "base_url": "https://api.z.ai/api/paas/v4/",
            "model": "glm-5",
        },
        "qwen3.5-397b": {
            "api_key": os.getenv("OPENAI_KEY", ""),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3.5-397b-a17b",
        },
        "deepseek-v3.2": {
            "api_key": os.getenv("OPENAI_KEY", ""),
            "base_url": os.getenv("VOLCANO_ENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            "model": "deepseek-v3-2-251201",
        },
        "kimi-k2.5": {
            "api_key": os.getenv("OPENAI_KEY", ""),
            "base_url": "https://api.moonshot.cn/v1",
            "model": "kimi-k2.5",
        },
        "minimax-m2.5": {
            "api_key": os.getenv("OPENAI_KEY", ""),
            "base_url": "https://api.minimaxi.com/v1",
            "model": "MiniMax-M2.5",
        },
        "aiXapply": {
            "api_key": os.getenv("OPENAI_KEY", "EMPTY"),
            "base_url": os.getenv("GENERATE_UDIFF_LOCAL_BASE_URL", "http://127.0.0.1:14123/v1"),
            "model": "aiXapply",
        },
        "custom": {},
    }

    preset = model_configs.get(args.provider, {})

    args.api_key = args.api_key if args.api_key is not None else preset.get("api_key", "")
    args.base_url = args.base_url if args.base_url is not None else preset.get("base_url", "")
    args.model = args.model if args.model is not None else preset.get("model", "")
    args.extra_body = _parse_extra_body(args.extra_body) or preset.get("extra_body", {})

    if args.model.startswith("kimi"):
        args.temperature = 0.6
        args.top_p = 0.95

    if not args.api_key:
        raise ValueError("api_key is empty. Provide --api-key or choose a provider preset.")
    if not args.base_url:
        raise ValueError("base_url is empty. Provide --base-url or choose a provider preset.")
    if not args.model:
        raise ValueError("model is empty. Provide --model or choose a provider preset.")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()

