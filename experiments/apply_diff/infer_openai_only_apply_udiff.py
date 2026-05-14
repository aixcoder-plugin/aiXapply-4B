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

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    # Optional dependency; environment variables can still be provided by the shell.
    pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]


PROMPT_UDIFF_APPLY = {
    "system": """You are a deterministic Code Patching Engine. Your task is to synthesize a "Updated File" by applying a partial "Update Snippet" to the provided "Source File".

### Algorithm
1. **Context Matching**: Analyze the `Update Snippet` to identify the context anchors (the lines of code surrounding the changes). Locate the exact corresponding block in the `Source File`. The match must be unique. The Update Snippet is a STANDARD UNIFIED DIFF (e.g. output of `git diff`), starting with lines like `--- a/...` and `+++ b/...`.
2. **Code Merging**: Replace the matched block in the `Source File` with the logic from the `Update Snippet`.
3. **Output Generation**: Output the FULL content of the resulting file.

### Constraints
- **NO Laziness**: Never output comments like `// ... rest of code ...` in the final output. You must write out every single line of the final code.
- **Strict Fidelity**: Preserve the original indentation style (spaces/tabs) and comments of the Source File for all unchanged parts.
- **Safety**: If the context in the snippet is ambiguous or cannot be found, output nothing inside the tags.

### Output Format
<update_file>[Your final code here]</update_file>
""".strip(),
    "user": """<language>{language}</language>

<source_file>{source_file}</source_file>

<update_snippet>{update_snippet}</update_snippet>

Please generate the full updated code strictly following the instructions.""",
}


PROMPT_SEARCH_REPLACE_APPLY = {
    "system": """You are a deterministic Code Patching Engine. Your task is to synthesize a "Updated File" by applying a partial "Update Snippet" to the provided "Source File".

### Algorithm
1. **Context Matching**: Analyze the `Update Snippet` to identify the context anchors (the lines of code surrounding the changes). Locate the exact corresponding block in the `Source File`. The match must be unique. The `Update Snippet` contains one or more SEARCH/REPLACE blocks delimited by `<<<<<<< SEARCH`, `=======`, and `>>>>>>> REPLACE`.
2. **Code Merging**: For each SEARCH/REPLACE block, replace the matched `SEARCH` block in the `Source File` with the corresponding `REPLACE` block, in order.
3. **Output Generation**: Output the FULL content of the resulting file.

### Constraints
- **NO Laziness**: Never output comments like `// ... rest of code ...` in the final output. You must write out every single line of the final code.
- **Strict Fidelity**: Preserve the original indentation style (spaces/tabs) and comments of the Source File for all unchanged parts.
- **Safety**: If any `SEARCH` block is ambiguous or cannot be found, output nothing inside the tags.

### Output Format
<update_file>[Your final code here]</update_file>
""".strip(),
    "user": """<language>{language}</language>

<source_file>{source_file}</source_file>

<update_snippet>{update_snippet}</update_snippet>

Please generate the full updated code strictly following the instructions.""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an apply model on a unified-diff dataset: send (original_code + udiff) to an "
            "OpenAI-compatible endpoint, collect full-file predictions, and write jsonl compatible "
            "with experiments/evaluation/run_evaluation.py."
        )
    )
    parser.add_argument(
        "-d",
        "--data-path",
        required=True,
        help="Path to the parquet dataset file (diff-xyz-test).",
    )
    parser.add_argument(
        "--diff-field",
        default="udiff",
        help="Which dataset column to use as update_snippet (default: udiff). e.g. udiff, udiff-h, udiff-l, search-replace",
    )
    parser.add_argument(
        "--filter-num",
        type=int,
        default=0,
        help="Only run the first N examples (0 means all).",
    )
    parser.add_argument(
        "-p",
        "--provider",
        default="local",
        choices=["VolcanoEngine", "local", "local2", "custom"],
        help="Provider preset. Use custom to rely on --api-key/--base-url/--model.",
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
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p nucleus sampling value.")
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
        default=2,
        help="Retry attempts for recoverable errors.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output jsonl path. Default: predictions/{model}_udiff_{timestamp}.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Do not call the model; write placeholder predictions (useful to validate the pipeline).",
    )
    return parser.parse_args()

def _add_line_numbers(text: str, *, prefix: str = "L") -> str:
    # Use \n as display newline; this is a reference view for the model only.
    lines = text.splitlines()
    return "\n".join(f"{prefix}{i + 1}:{line}" for i, line in enumerate(lines))

def build_messages(
    *,
    language: str,
    source_file: str,
    update_snippet: str,
    diff_field: str = "udiff",
) -> List[Dict[str, Any]]:
    prompt = PROMPT_SEARCH_REPLACE_APPLY if diff_field == "search-replace" else PROMPT_UDIFF_APPLY
    return [
        {"role": "system", "content": prompt["system"]},
        {
            "role": "user",
            "content": prompt["user"].format(
                language=language,
                source_file=source_file,
                update_snippet=update_snippet,
            ),
        },
    ]


async def generate_completion(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    messages: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Optional[str], float]:
    attempt = 0
    backoff_factor = 2

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
            logger.info(
                "Retrying in %s seconds (attempt %s/%s)",
                wait_time,
                attempt,
                args.max_retries,
            )
            await asyncio.sleep(wait_time)
        else:
            return None, 0.0
    return None, 0.0


def build_record(
    *,
    idx: int,
    example: Dict[str, Any],
    prediction_raw: Optional[str],
    elapsed_time: float,
    diff_field: str,
) -> Dict[str, Any]:
    language = str(example.get("lang") or example.get("language") or "")
    original_code = str(example.get("old_code") or example.get("original_code") or "")
    ground_truth = str(example.get("new_code") or example.get("final_code") or "")
    update_snippet = str(example.get(diff_field) or "")

    # Keep prediction as the RAW model output (for downstream error classification that checks wrapper).
    prediction = prediction_raw if isinstance(prediction_raw, str) else ""
    if len(prediction.strip()) == 0:
        # experiments/evaluation/run_evaluation.py requires a non-empty prediction string.
        prediction = "<update_file></update_file>"

    return {
        # Required keys for experiments/evaluation/run_evaluation.py
        "index": int(idx),
        "prediction": prediction,
        "elapsed_time": float(elapsed_time),
        "ground_truth": ground_truth,
        "original_code": original_code,
        "update_snippet": update_snippet,
        "language": language,
        # Extra metadata for debugging / analysis
        "repo": example.get("repo"),
        "commit": example.get("commit"),
        "path": example.get("path"),
        "message": example.get("message"),
        "license": example.get("license"),
        "change_kind": example.get("change_kind"),
        "n_added": example.get("n_added"),
        "n_removed": example.get("n_removed"),
        "n_hunks": example.get("n_hunks"),
        "diff_field": diff_field,
    }


async def process_example(
    idx: int,
    example: Dict[str, Any],
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Any]]:
    language = str(example.get("lang") or "")
    file_path = str(example.get("path") or "")
    commit_message = str(example.get("message") or "")
    source_file = str(example.get("old_code") or "")
    update_snippet = str(example.get(args.diff_field) or "")

    # Dry-run: no model call, produce a deterministic placeholder prediction.
    if args.dry_run:
        prediction_raw = f"<update_file>{source_file}</update_file>"
        return idx, build_record(
            idx=idx,
            example=example,
            prediction_raw=prediction_raw,
            elapsed_time=0.0,
            diff_field=args.diff_field,
        )

    messages = build_messages(
        language=language,
        source_file=source_file,
        update_snippet=update_snippet,
        diff_field=args.diff_field,
    )
    prediction_raw, elapsed_time = await generate_completion(client, limiter, messages, args)
    # If model returns unwrapped content, keep it as-is (classification will label it OUTPUT_INVALID).
    # If model returns wrapped content, keep it raw; evaluation strips tags when scoring.
    return idx, build_record(
        idx=idx,
        example=example,
        prediction_raw=prediction_raw,
        elapsed_time=elapsed_time,
        diff_field=args.diff_field,
    )



def _load_dataset_parquet(data_path: Path):
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")
    if data_path.is_file() and data_path.suffix.lower() == ".parquet":
        return load_dataset("parquet", data_files={"data": str(data_path)})["data"]
    # fallback: datasets-compatible path
    return load_dataset(str(data_path))["train"]


def _parse_extra_body(extra_body: Optional[str]) -> Dict[str, Any]:
    if not extra_body:
        return {}
    try:
        obj = json.loads(extra_body)
    except Exception as e:
        raise ValueError(f"--extra-body must be valid JSON. got error: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("--extra-body must be a JSON object.")
    return obj


async def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    dataset = _load_dataset_parquet(data_path)

    if args.filter_num > 0:
        dataset = dataset.select(range(min(args.filter_num, len(dataset))))
        logger.info("Filtered dataset to %s examples", len(dataset))

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = SCRIPT_DIR / "predictions" / f"{args.model}_{args.diff_field}_{run_timestamp}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("output_path:", str(output_path), flush=True)

    limiter = AsyncLimiter(args.rate_limit, args.rate_period)
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    progress = tqdm(total=len(dataset), desc="Requesting(UDIFF)")
    buffer: Dict[int, Dict[str, Any]] = {}
    next_to_write = 0
    pending: set[asyncio.Task] = set()
    task_to_index: Dict[asyncio.Task, int] = {}
    index_iter = iter(range(len(dataset)))

    def enqueue() -> bool:
        try:
            index = next(index_iter)
        except StopIteration:
            return False
        task = asyncio.create_task(process_example(index, dataset[index], client, limiter, args))
        pending.add(task)
        task_to_index[task] = int(index)
        return True

    for _ in range(min(args.concurrency, len(dataset))):
        enqueue()

    with output_path.open("w", encoding="utf-8") as out_file:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                idx_hint = task_to_index.pop(task, None)
                try:
                    idx, record = task.result()
                except Exception as e:
                    # Ensure output is still ordered and complete: write a placeholder record.
                    logger.exception("Task failed at idx=%s", idx_hint)
                    if idx_hint is None:
                        progress.update(1)
                        continue
                    try:
                        example = dataset[int(idx_hint)]
                    except Exception:
                        example = {}
                    record = build_record(
                        idx=int(idx_hint),
                        example=example,
                        prediction_raw="<update_file></update_file>",
                        elapsed_time=0.0,
                        diff_field=args.diff_field,
                    )
                    record["inference_error"] = repr(e)
                    idx = int(idx_hint)

                buffer[idx] = record
                while next_to_write in buffer:
                    current = buffer.pop(next_to_write)
                    out_file.write(json.dumps(current, ensure_ascii=False) + "\n")
                    out_file.flush()
                    next_to_write += 1
                progress.update(1)

            while len(pending) < args.concurrency and enqueue():
                continue

    progress.close()
    logger.info("UDIFF predictions written to %s", output_path)


def main() -> None:
    args = parse_args()

    model_configs = {
        "VolcanoEngine": {
            "api_key": os.getenv("VOLCANO_ENGINE_API_KEY", ""),
            "base_url": os.getenv("VOLCANO_ENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            "model": "deepseek-v3-2-251201",
        },
        "local": {
            "api_key": os.getenv("LOCAL_OPENAI_API_KEY", "EMPTY"),
            "base_url": os.getenv("APPLY_DIFF_LOCAL_BASE_URL", "http://127.0.0.1:14123/v1"),
            "model": "aiXapply",
        },
        "local2": {
            "api_key": os.getenv("LOCAL_OPENAI_API_KEY", "EMPTY"),
            "base_url": os.getenv("APPLY_DIFF_LOCAL2_BASE_URL", "http://127.0.0.1:14124/v1"),
            "model": "qwen-4b",
        },
        "custom": {},
    }

    preset = model_configs.get(args.provider, {})

    # Resolve connection settings: CLI overrides preset.
    args.api_key = args.api_key if args.api_key is not None else preset.get("api_key", "")
    args.base_url = args.base_url if args.base_url is not None else preset.get("base_url", "")
    args.model = args.model if args.model is not None else preset.get("model", "")
    args.extra_body = _parse_extra_body(args.extra_body) or preset.get("extra_body", {})

    if args.provider == "VolcanoEngine" and not args.api_key:
        raise ValueError(
            "Missing credential for provider VolcanoEngine. "
            "Set VOLCANO_ENGINE_API_KEY or pass --api-key."
        )
    if not args.base_url:
        raise ValueError("base_url is empty. Provide --base-url or choose a provider preset.")
    if not args.model:
        raise ValueError("model is empty. Provide --model or choose a provider preset.")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()

