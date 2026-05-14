"""
aiXapply data generation pipeline.

Stage: multi-model inference verification.
Purpose: run generated prompts against VolcanoEngine or bailian endpoints.
Inputs: per-language passed parquet datasets.
Outputs: per-provider prediction jsonl files.
"""
import argparse
import asyncio
import glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from aiolimiter import AsyncLimiter
from aiohttp import ClientError, ClientResponseError
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from prompt import PROMPT_TRAIN

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference against an OpenAI-compatible endpoint using the prompts "
            "stored in the dataset (similar to run_inference.py, but via API calls)."
        )
    )
    parser.add_argument(
        "-d","--data-dir",
        default="dataset/synthetic_data/judge_processed",
        help="Path to the directory containing language parquet files (e.g., python_passed.parquet).",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Specify languages to process (default: process all *_passed.parquet files).",
    )
    parser.add_argument(
        "--filter-num",
        type=int,
        default=0,
        help="Filter the dataset by the number of examples.",
    )
    parser.add_argument(
        "-p","--provider",
        default="bailian",
        choices=["VolcanoEngine", "bailian"],
        help="Provider name",
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
        default=0.5,
        help="Top-p nucleus sampling value.",
    )
    parser.add_argument(
        "-c","--concurrency",
        type=int,
        default=64,
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
        "-o", "--output-dir",
        default="dataset/synthetic_data/predictions",
        help="Output directory for prediction files.",
    )
    parser.add_argument(
        "-f", "--frequency",
        default=1,
        help="Frequency of prediction files.",
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
        except ClientResponseError as cre:
            status = cre.status
            logger.error("HTTP error %s: %s", status, cre.message)
            if status not in {429, 500, 503}:
                return None, 0.0
        except ClientError as ce:
            logger.error("Client error: %s", ce)
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


def build_messages_from_example(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build messages from data processed by judge_processor.py.
    
    Data format:
    - original_code: The original code
    - update_snippet: The update snippet
    - final_code: The final code (ground truth)
    """
    original_code = example.get("original_code", "")
    update_snippet = example.get("update_snippet", "")
    language = example.get("language")
    # If the data already has a prompt field (from verl_processor), use it directly
    if "prompt" in example and example["prompt"]:
        prompt = example["prompt"]
        if isinstance(prompt, list) and len(prompt) > 0:
            return prompt
    
    # Otherwise build messages from original_code and update_snippet
    if not original_code or not update_snippet:
        return []
    
    messages = [
        {
            "role": "system",
            "content": PROMPT_TRAIN['system'],
        },
        {
            "role": "user",
            "content": PROMPT_TRAIN['user'].format(
                language=language,
                source_file=original_code, 
                update_snippet=update_snippet
            ),
        }
    ]
    return messages


def build_record(
    idx: int, example: Dict[str, Any], prediction: Optional[str], elapsed_time: float
) -> Dict[str, Any]:
    original_code = example.get("original_code", "")
    update_snippet = example.get("update_snippet", "")
    
    # ground_truth may be in reward_model or directly in final_code
    reward_model = example.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth") or example.get("final_code", "")

    return {
        "index": int(idx),
        "prediction": prediction,
        "elapsed_time": elapsed_time,
        "ground_truth": ground_truth,
        "original_code": original_code,
        "update_snippet": update_snippet,
    }


async def process_example(
    idx: int,
    example: Dict[str, Any],
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Any]]:
    messages = build_messages_from_example(example)
    
    if not messages:
        logger.warning(f"Empty messages at index {idx}, skipping")
        return idx, build_record(idx, example, None, 0.0)

    prediction, elapsed_time = await generate_completion(client, limiter, messages, args)

    record = build_record(idx, example, prediction, elapsed_time)
    return idx, record


async def run_single_language(
    data_path: Path,
    language: str,
    output_path: Path,
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    args: argparse.Namespace
) -> None:
    """Process a single language data file for prediction."""
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("parquet", data_files={'data': str(data_path)})['data']

    if args.filter_num > 0:
        dataset = dataset.select(range(min(args.filter_num, len(dataset))))
        print("================================================")
        print("dataset: ", dataset[0])
        print("================================================")

    print(f"[{language}] data_path: {data_path}")
    print(f"[{language}] output_path: {output_path}")
    print(f"[{language}] dataset size: {len(dataset)}")

    progress = tqdm(total=len(dataset), desc=f"[{language}] Requesting")
    buffer: Dict[int, Dict[str, Any]] = {}
    next_to_write = 0
    pending: set[asyncio.Task] = set()
    index_iter = iter(range(len(dataset)))
    skipped_indices = []

    def enqueue() -> bool:
        index = -1
        while index == -1 or index in skipped_indices:
            try:
                index = next(index_iter)
            except StopIteration:
                return False
        
        print(f"[{language}] Start request! idx: {index}", flush=True)

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
                    # Add language identifier to the record
                    record['language'] = language
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
                    print(f"[{language}] Save success! idx: {current['index']}", flush=True)
                    next_to_write += 1
                progress.update(1)

            while len(pending) < args.concurrency and enqueue():
                continue
    progress.close()
    logger.info("[%s] Predictions written to %s", language, output_path)


def discover_language_files(data_dir: str, languages: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
    """
    Discover language files in the data directory.
    
    Args:
        data_dir: Data directory
        languages: Specified language list, None means auto-discovery
    
    Returns:
        List of (language name, file path)
    """
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all *_passed.parquet files generated by judge_processor.py.
    pattern = str(data_dir_path / "*_passed.parquet")
    all_files = glob.glob(pattern)
    
    result = []
    for file_path in sorted(all_files):
        # Extract language name from file name, e.g., python_passed.parquet -> python
        file_name = os.path.basename(file_path)
        lang = file_name.replace("_passed.parquet", "")
        
        # If specified language list, only process specified languages
        if languages is not None and lang not in languages:
            continue
        
        result.append((lang, Path(file_path)))
    
    return result


async def run(args: argparse.Namespace) -> None:
    """Main running function, process all language files."""
    # Discover language files
    language_files = discover_language_files(args.data_dir, args.languages)
    
    if not language_files:
        logger.error(f"No language files found in {args.data_dir}")
        return
    
    logger.info(f"Found {len(language_files)} language files to process:")
    for lang, file_path in language_files:
        logger.info(f"- {lang}: {file_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create shared client and limiter
    limiter = AsyncLimiter(args.rate_limit, args.rate_period)
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    
    # Save all output paths
    all_output_paths: List[Tuple[str, Path, bool]] = []  # (language, path, success)
    
    # Process each language sequentially
    for lang, data_path in tqdm(language_files, desc="Processing languages"):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing language: {lang}")
        logger.info(f"{'='*60}")
        
        output_path = output_dir / f"{args.model}_{lang}_{args.frequency}.jsonl"
        
        try:
            await run_single_language(
                data_path=data_path,
                language=lang,
                output_path=output_path,
                client=client,
                limiter=limiter,
                args=args
            )
            all_output_paths.append((lang, output_path, True))
        except Exception as e:
            logger.exception(f"[{lang}] Failed to process: {e}")
            all_output_paths.append((lang, output_path, False))
            continue
    
    # Print all output paths summary
    print("\n" + "=" * 80)
    print("Output file summary")
    print("=" * 80)
    print(f"{'Language':<20} {'Status':<10} {'Output Path'}")
    print("-" * 80)
    
    success_count = 0
    for lang, output_path, success in all_output_paths:
        status = "Success" if success else "Failed"
        if success:
            success_count += 1
        print(f"{lang:<20} {status:<10} {output_path}")
    
    print("-" * 80)
    print(f"Total: {success_count}/{len(all_output_paths)} languages processed successfully")
    print("=" * 80)
    
    logger.info("\nAll languages processed!")

def main() -> None:
    args = parse_args()

    model_configs = {
        "VolcanoEngine": {
            "api-key": os.getenv("VOLCANO_ENGINE_API_KEY", ""),
            "base-url": os.getenv("VOLCANO_ENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            "model": os.getenv("VOLCANO_ENGINE_MODEL", "deepseek-v3-2-251201"),
        },
        "bailian": {
            "api-key": os.getenv("BAILIAN_API_KEY", ""),
            "base-url": os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "model": os.getenv("BAILIAN_MODEL", "glm-4.7"),
            "extra_body": {
                "enable_thinking": False
            }
        }
    }
    args.api_key = model_configs[args.provider]["api-key"]
    args.base_url = model_configs[args.provider]["base-url"]
    args.model = model_configs[args.provider]["model"]
    args.extra_body = model_configs[args.provider].get("extra_body",{})

    if not args.api_key:
        raise ValueError(
            f"Missing credential for provider {args.provider}. "
            "Set the matching environment variable before running this script."
        )
    
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
