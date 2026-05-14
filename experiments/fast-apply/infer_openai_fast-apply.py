"""
Run inference for a single AI model on the test dataset.
Generate a complete jsonl file containing prediction and elapsed_time.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tqdm import tqdm

from data_generation.prompt import PROMPT_TRAIN


# NOTE: The Fast-Apply system and user prompts below are reused verbatim from
# the upstream Fast-Apply project so that the Fast-Apply-7B baseline runs under
# its native prompting. See https://github.com/kortix-ai/fast-apply (Apache-2.0)
# and the matching entry in the repository NOTICE file.
FASTAPPLY_SYSTEM_PROMPT = (
    "You are a coding assistant that helps merge code updates, ensuring every "
    "modification is fully integrated."
)


FASTAPPLY_USER_PROMPT = """Merge all changes from the <update> snippet into the <code> below.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, ellipses, or code fences.

<code>{original_code}</code>

<update>{update_snippet}</update>

Provide the complete updated code."""


def build_prompt_messages(sample: Dict[str, Any], model_name: str) -> list[Dict[str, str]]:
    is_fastapply = "fastapply" in model_name.lower()

    if is_fastapply:
        return [
            {"role": "system", "content": FASTAPPLY_SYSTEM_PROMPT},
            {"role": "user", "content": FASTAPPLY_USER_PROMPT.format(
                original_code=sample["original_code"],
                update_snippet=sample["update_snippet"]
            )}
        ]

    return [
        {"role": "system", "content": PROMPT_TRAIN["system"]},
        {"role": "user", "content": PROMPT_TRAIN["user"].format(
            language=sample["language"],
            source_file=sample["original_code"],
            update_snippet=sample["update_snippet"]
        )}
    ]


def _normalize_sample(raw: Dict[str, Any], fallback_index: int) -> Dict[str, Any]:
    extra_info = raw.get("extra_info")
    reward_model = raw.get("reward_model")

    if not isinstance(extra_info, dict):
        raise KeyError(f"Sample is missing extra_info or it has the wrong type; index={fallback_index}")
    if not isinstance(reward_model, dict):
        raise KeyError(f"Sample is missing reward_model or it has the wrong type; index={fallback_index}")

    required_extra = ["language", "original_code", "update_snippet"]
    missing_extra = [field for field in required_extra if field not in extra_info]
    if missing_extra:
        raise KeyError(f"extra_info is missing required fields: {missing_extra}; index={fallback_index}")
    if "ground_truth" not in reward_model:
        raise KeyError(f"reward_model is missing required field: ['ground_truth']; index={fallback_index}")

    return {
        "index": int(extra_info.get("index", fallback_index)),
        "language": str(extra_info["language"]),
        "original_code": str(extra_info["original_code"]),
        "update_snippet": str(extra_info["update_snippet"]),
        "ground_truth": str(reward_model["ground_truth"]),
    }


def load_samples(input_path: str) -> list[Dict[str, Any]]:
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with open(path, 'r', encoding='utf-8-sig') as f:
            raw_samples = [json.loads(line) for line in f if line.strip()]
        return [_normalize_sample(sample, i) for i, sample in enumerate(raw_samples)]

    if suffix == ".parquet":
        import pyarrow.parquet as pq  # type: ignore
        table = pq.read_table(path)
        rows = table.to_pylist()

        return [_normalize_sample(dict(row), i) for i, row in enumerate(rows)]

    raise ValueError(f"Unsupported input format: {suffix}. Only .jsonl and .parquet are supported.")


async def generate_completion(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    messages: list,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    max_retries: int
) -> Tuple[Optional[str], float]:
    """Call the API and generate a completion."""
    attempt = 0
    backoff_factor = 2
    
    while attempt <= max_retries:
        try:
            async with limiter:
                start_time = time.time()
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stream=False,
                    ),
                    timeout=timeout,
                )
                content = response.choices[0].message.content
                elapsed = time.time() - start_time
                return content, elapsed
        except asyncio.TimeoutError:
            print(f"[WARN] Request timed out (attempt {attempt + 1}/{max_retries + 1})")
        except Exception as e:
            print(f"[ERROR] API call failed: {e} (attempt {attempt + 1}/{max_retries + 1})")
        
        attempt += 1
        if attempt <= max_retries:
            wait_time = backoff_factor ** attempt
            await asyncio.sleep(wait_time)
    
    return None, 0.0


async def process_sample(
    idx: int,
    sample: Dict[str, Any],
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    max_retries: int,
) -> Dict[str, Any]:
    """Process a single sample."""
    messages = build_prompt_messages(sample, model_name)
    
    prediction, elapsed_time = await generate_completion(
        client, limiter, messages, model_name,
        max_tokens, temperature, top_p, timeout, max_retries
    )

    # Preserve the original record and attach prediction metadata.
    result = sample.copy()
    result["prediction"] = prediction or ""
    result["elapsed_time"] = elapsed_time
    
    return result


async def run_inference(args):
    """Run inference."""
    print("=" * 60)
    print("[INFER] Starting test-set inference")
    print("=" * 60)
    
    # Load input samples.
    print(f"[INPUT] {args.input}")
    samples = load_samples(args.input)
    
    print(f"[INFO] Total samples: {len(samples)}")
    print(f"[MODEL] {args.model}")
    print(f"[CONCURRENCY] {args.concurrency}")
    print(f"[RATE] {args.rate_limit} requests/minute")
    
    # Initialize the client.
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    limiter = AsyncLimiter(args.rate_limit, args.rate_period)
    
    # Prepare the output file.
    if args.output:
        output_file = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = args.model.replace("/", "_").replace(":", "_")
        output_file = f"predictions/{model_name}_{timestamp}.jsonl"
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    print(f"[OUTPUT] {output_file}")
    
    # Process requests in batches.
    results = []
    
    print("\n[PROCESS] Running inference...")
    with tqdm(total=len(samples), desc="Inference progress") as pbar:
        for i in range(0, len(samples), args.concurrency):
            batch = samples[i:i + args.concurrency]
            batch_tasks = [
                process_sample(
                    sample["index"], sample, client, limiter,
                    args.model, args.max_tokens, args.temperature,
                    args.top_p, args.timeout, args.max_retries,
                )
                for sample in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            pbar.update(len(batch))
    
    # Save results sorted by index.
    print("\n[SAVE] Writing result file...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in sorted(results, key=lambda x: x["index"]):
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"[DONE] Saved to: {output_file}")
    
    # Summarize the run.
    success_count = sum(1 for r in results if r["prediction"])
    total_time = sum(r["elapsed_time"] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    print("\n[SUMMARY]")
    print(f"  Success: {success_count}/{len(results)}")
    print(f"  Failed: {len(results) - success_count}/{len(results)}")
    print(f"  Avg time: {avg_time:.2f}s/sample")
    print(f"  Total time: {total_time:.2f}s")
    
    print("\n[NEXT]")
    print(f"  Run evaluation: python experiments/evaluation/run_evaluation.py -i {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Run model inference on the test dataset."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input test file. Supports .jsonl/.parquet."
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: predictions/MODEL_TIMESTAMP.jsonl)."
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Endpoint credential."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="API base URL (for example: https://api.openai.com/v1)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (for example: gpt-4, deepseek-chat)."
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=10,
        help="Concurrent request count (default: 10)."
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=60,
        help="Rate limit in requests per minute (default: 60)."
    )
    parser.add_argument(
        "--rate-period",
        type=int,
        default=60,
        help="Rate-limit window in seconds (default: 60)."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128000,
        help="Maximum generated tokens (default: 128000)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)."
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p sampling value (default: 0.8)."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds (default: 300)."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry count (default: 3)."
    )
    
    args = parser.parse_args()
    
    # Run inference.
    asyncio.run(run_inference(args))


if __name__ == "__main__":
    main()
