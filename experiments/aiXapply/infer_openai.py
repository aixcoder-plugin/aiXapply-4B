import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from aiolimiter import AsyncLimiter
from datasets import load_dataset
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    # Optional dependency; environment variables can still be provided by the shell.
    pass

from data_generation.prompt import PROMPT_TRAIN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

from datetime import datetime
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference against an OpenAI-compatible endpoint using the prompts "
            "stored in the dataset (similar to run_inference.py, but via API calls)."
        )
    )
    parser.add_argument(
        "-d","--data-path",
        required=True,
        help="Path to the evaluation parquet file.",
    )
    parser.add_argument(
        "--filter-num",
        type=int,
        default=0,
        help="Filter the dataset by the number of examples.",
    )
    parser.add_argument(
        "-p","--provider",
        default="local",
        choices=[ "VolcanoEngine","local"],
        help="Provider name, choices: VolcanoEngine, local.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32*1024,
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
        "-c","--concurrency",
        type=int,
        default=16,
        help="Maximum concurrent requests to send.",
    )
    parser.add_argument(
        "-r","--rate-limit",
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
        "--resume",
        type=str,
        default=None,
        help="Path to an existing predictions .jsonl file. "
             "Null-prediction entries will be re-inferred (streaming) "
             "and the file will be updated in-place.",
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
                stream = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=args.model,
                        stream=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        messages=messages,
                        extra_body=args.extra_body or {},
                    ),
                    timeout=args.request_timeout,
                )
                chunks: List[str] = []
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)
                end_time = time.time()
            content = "".join(chunks) if chunks else None
            return content, end_time - start_time
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


def build_record(
    idx: int, example: Dict[str, Any], prediction: Optional[str], elapsed_time: float, language: str
) -> Dict[str, Any]:
    messages = example.get("prompt", [])
    user_prompt = extract_user_content(messages)
    original_code, update_snippet = extract_tagged_blocks(user_prompt)
    reward_model = example.get("reward_model") or {}

    return {
        "index": int(idx),
        "prediction": prediction,
        "elapsed_time": elapsed_time,
        "ground_truth": reward_model.get("ground_truth"),
        "original_code": original_code,
        "update_snippet": update_snippet,
        "language": language,
    }


async def process_example(
    idx: int,
    example: Dict[str, Any],
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Any]]:
    messages = example.get("prompt") or []
    if not isinstance(messages, list):
        raise ValueError(f"Unexpected prompt format at index {idx}: {messages!r}")

    prediction, elapsed_time = await generate_completion(client, limiter, messages, args)
    record = build_record(idx, example, prediction, elapsed_time, example['extra_info']['language'])
    return idx, record


async def run_resume(args: argparse.Namespace) -> None:
    """Resume mode: re-infer only null-prediction entries in an existing file."""
    resume_path = Path(args.resume)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume file not found: {resume_path}")

    existing_records: List[Dict[str, Any]] = []
    with resume_path.open("r", encoding="utf-8") as f:
        for line in f:
            existing_records.append(json.loads(line))

    null_indices: List[int] = []
    record_map: Dict[int, Dict[str, Any]] = {}
    for rec in existing_records:
        idx = rec["index"]
        record_map[idx] = rec
        if rec.get("prediction") is None:
            null_indices.append(idx)

    logger.info(
        "Loaded %d records from %s, %d have null predictions to re-infer",
        len(existing_records), resume_path, len(null_indices),
    )
    if not null_indices:
        logger.info("Nothing to resume – all predictions are present.")
        return

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")

    dataset = load_dataset("parquet", data_files={'data': str(data_path)})['data']

    if args.filter_num > 0:
        dataset = dataset.select(range(min(args.filter_num, len(dataset))))

    limiter = AsyncLimiter(args.rate_limit, args.rate_period)
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    progress = tqdm(total=len(null_indices), desc="Re-inferring null predictions")
    pending: set[asyncio.Task] = set()
    index_iter = iter(null_indices)
    filled_count = 0

    def enqueue() -> bool:
        try:
            index = next(index_iter)
        except StopIteration:
            return False

        logger.debug("Start re-infer: idx=%s", index)

        task = asyncio.create_task(
                process_example(index, dataset[index], client, limiter, args)
        )
        pending.add(task)
        return True

    for _ in range(min(args.concurrency, len(null_indices))):
        enqueue()

    while pending:
        done, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            try:
                idx, record = task.result()
            except Exception:
                logger.exception("Task failed for re-infer")
                progress.update(1)
                continue

            if record.get("prediction") is not None:
                record_map[idx] = record
                filled_count += 1
                logger.debug("Re-infer success: idx=%s", idx)
            else:
                logger.warning("Re-infer still null for idx %d", idx)
            progress.update(1)

        while len(pending) < args.concurrency and enqueue():
            continue

    progress.close()
    logger.info("Re-inferred %d / %d null predictions", filled_count, len(null_indices))

    sorted_indices = sorted(record_map.keys())
    tmp_path = resume_path.with_suffix(resume_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as out_file:
        for idx in sorted_indices:
            out_file.write(json.dumps(record_map[idx], ensure_ascii=False) + "\n")
        out_file.flush()
        os.fsync(out_file.fileno())
    tmp_path.replace(resume_path)
    logger.info("Updated predictions written back to %s", resume_path)


async def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")

    output_path = REPO_ROOT / "predictions" / f"{args.model}_{run_timestamp}.jsonl"
    logger.info("Output path: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("parquet", data_files={"data": str(data_path)})["data"]

    if args.filter_num > 0:
        dataset = dataset.select(range(min(args.filter_num, len(dataset))))
        logger.debug("First example after filter: %s", dataset[0])

    logger.info("Dataset path: %s", data_path)
    logger.info("Dataset: %s", dataset)

    limiter = AsyncLimiter(args.rate_limit, args.rate_period)
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    progress = tqdm(total=len(dataset), desc="Requesting")
    buffer: Dict[int, Dict[str, Any]] = {}
    next_to_write = 0
    pending: set[asyncio.Task] = set()
    index_iter = iter(range(len(dataset)))

    skipped_indices: list[int] = []
    logger.debug("skipped_indices=%s", skipped_indices)

    def enqueue() -> bool:
        index = -1
        while index == -1 or index in skipped_indices:
            try:
                index = next(index_iter)
            except StopIteration:
                return False

        logger.debug("Start request: idx=%s", index)

        task = asyncio.create_task(
                process_example(index, dataset[index], client, limiter, args)
        )
        pending.add(task)
        return True

    for _ in range(min(args.concurrency, len(dataset))):
        enqueue()

    with output_path.open("w", encoding="utf-8") as out_file:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
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
                    logger.debug("Save success: idx=%s", current["index"])
                    next_to_write += 1
                progress.update(1)

            while len(pending) < args.concurrency and enqueue():
                continue
    progress.close()
    logger.info("Predictions written to %s", output_path)

def main() -> None:
    args = parse_args()

    model_configs = {
    "VolcanoEngine": {
        "api-key": os.getenv("VOLCANO_ENGINE_API_KEY", ""),
        "base-url": os.getenv("VOLCANO_ENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        "model": "deepseek-v3-2-251201"
    },
    "local": {
        "api-key": os.getenv("LOCAL_OPENAI_API_KEY", ""),
        "base-url": os.getenv("LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:12003/v1"),
        "model": os.getenv("LOCAL_OPENAI_MODEL", "aiXapply-4B-RL"),
    }
    }
    args.api_key = model_configs[args.provider]["api-key"]
    args.base_url = model_configs[args.provider]["base-url"]
    args.model = model_configs[args.provider]["model"]
    args.extra_body = model_configs[args.provider].get("extra_body",{})

    if args.provider == "VolcanoEngine" and not args.api_key:
        raise ValueError(
            "Missing credential for provider VolcanoEngine. "
            "Set VOLCANO_ENGINE_API_KEY before running this script."
        )
    if args.resume:
        asyncio.run(run_resume(args))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
