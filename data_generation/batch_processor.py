"""
aiXapply data generation pipeline.

Stage: verification batch preparation.
Purpose: build LLM-as-judge requests from generated code results.
Inputs: code generation requests and code output_batch_*.jsonl files.
Outputs: verification batch_*.jsonl files and comparison artifacts.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import tiktoken
from tqdm import tqdm

from prompt import PROMPT_VERIFICATION
from utils import (
    EXTENSION_MAP,
    calculate_cost,
    update_data_pair,
    write_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============== Utility Functions ==============

def get_file_extension(lang: str) -> str:
    """Get file extension based on language type."""
    return EXTENSION_MAP.get(lang.lower(), lang.lower())


def extract_update_snippet(content: str) -> str:
    """Extract update_snippet from response content."""
    try:
        matches = re.findall(
            r"<update_snippet>(.*?)</update_snippet>",
            content,
            flags=re.DOTALL
        )
        return "".join(matches).strip() if matches else "ERROR"
    except (IndexError, AttributeError):
        return "ERROR"


def extract_final_code(content: str) -> str:
    """Extract final_code from response content."""
    try:
        return content.split("<final_code>")[1].split("</final_code>")[0].strip()
    except (IndexError, AttributeError):
        return "ERROR"


def save_code_files(
    compared_dir: str | Path,
    lang: str,
    task_num: str,
    update_snippet: str,
    final_code: str
) -> None:
    """Save update_snippet and final_code to compared directory."""
    ext = get_file_extension(lang)
    output_path = Path(compared_dir) / lang / str(task_num)
    output_path.mkdir(parents=True, exist_ok=True)
    
    (output_path / f"update_snippet.{ext}").write_text(
        update_snippet,
        encoding="utf-8"
    )
    (output_path / f"final_code.{ext}").write_text(
        final_code,
        encoding="utf-8"
    )


def add_line_numbers(code: str) -> str:
    """Add line numbers to code."""
    lines = code.splitlines()
    max_width = len(str(len(lines)))
    return '\n'.join([
        f"{i:{max_width}} | {line}"
        for i, line in enumerate(lines, start=1)
    ])


def generate_diff(original_code: str, final_code: str) -> str:
    """Generate unified diff between two code segments."""
    diff = difflib.unified_diff(
        original_code.splitlines(),
        final_code.splitlines(),
        n=3
    )
    return '\n'.join(diff)


# ============== Core Processing Logic ==============

def load_and_pair_batches(
    batch_dir: str,
    codes_dir: str,
    output_dir: str,
    compared_dir: str,
    language_type: str,
    tokenizer: tiktoken.Encoding
) -> tuple[int, int, int]:
    """
    Load batch files and pair code generation results.
    
    Args:
        batch_dir: Original code batch directory (contains input requests)
        codes_dir: Code generation result directory (contains output results)
        output_dir: Verification request output directory
        compared_dir: Compared file directory
        language_type: Language type
        tokenizer: Tokenizer
    
    Returns:
        (batch number, total input tokens, total max tokens)
    """
    # Find input batch files
    input_files = sorted(glob.glob(os.path.join(batch_dir, "batch_*.jsonl")))
    
    if not input_files:
        logger.warning(f"[{language_type}] No input batch files found: {batch_dir}")
        return 0, 0, 0
    
    total_input_tokens = 0
    total_max_tokens = 0
    file_count = 0
    
    system_prompt = PROMPT_VERIFICATION["system"]
    system_tokens = len(tokenizer.encode(system_prompt))
    
    ext = get_file_extension(language_type)
    
    for input_file in tqdm(
        input_files,
        desc=f"[{language_type}] Processing batch files",
        unit="file"
    ):
        data_pair = {}
        current_batch = []
        current_batch_tokens = 0
        
        # Parse batch number
        file_num = int(os.path.basename(input_file).split("_")[-1].split(".")[0])
        codes_file = os.path.join(codes_dir, f"output_batch_{file_num:03d}.jsonl")
        
        # Check if code generation result file exists
        if not os.path.exists(codes_file):
            logger.warning(f"[{language_type}] Code generation result file not found: {codes_file}")
            continue
        
        file_count += 1
        
        # Read input and code generation result files
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(codes_file, 'r', encoding='utf-8') as f_codes:
            
            # First step: process code generation results
            for line_codes in f_codes:
                data_codes = json.loads(line_codes)
                custom_id_out = data_codes['custom_id']
                
                try:
                    raw_content = data_codes['response']['body']['choices'][0]['message']['content']
                except (KeyError, IndexError):
                    logger.error(f"[{language_type}] Failed to parse code response: {custom_id_out}")
                    continue
                
                update_snippet = extract_update_snippet(raw_content)
                final_code = extract_final_code(raw_content)
                
                if update_snippet == "ERROR" or final_code == "ERROR":
                    logger.error(f"[{language_type}] Failed to extract code: {custom_id_out}")
                
                # Save to compared directory
                task_num = custom_id_out.split("-")[-1]
                save_code_files(
                    compared_dir, language_type, task_num,
                    update_snippet, final_code
                )
                
                data_pair = update_data_pair(data_pair, custom_id_out, {
                    'update_snippet': update_snippet,
                    'final_code': final_code,
                })
            
            # Second step: process input data
            for line_in in f_in:
                data_in = json.loads(line_in)
                custom_id_in = data_in['custom_id']
                user_prompt_in = data_in['user_prompt']
                
                try:
                    original_code = user_prompt_in.split("<original_code>")[1].split("</original_code>")[0].strip()
                except (IndexError, AttributeError):
                    logger.error(f"[{language_type}] Failed to parse original code: {custom_id_in}")
                    continue
                
                data_pair = update_data_pair(data_pair, custom_id_in, {
                    'original_code': original_code,
                    'request': data_in
                })
        
        # Third step: generate verification requests
        for custom_id, data in tqdm(
            data_pair.items(),
            desc=f"[{language_type}] Generating verification requests",
            unit="pair",
            leave=False
        ):
            original_code = data.get('original_code')
            update_snippet = data.get('update_snippet')
            final_code = data.get('final_code')
            
            if not all([original_code, update_snippet, final_code]):
                logger.warning(f"[{language_type}] Data is incomplete: {custom_id}")
                continue
            
            if update_snippet == "ERROR" or final_code == "ERROR":
                logger.warning(f"[{language_type}] Code extraction error, skipping: {custom_id}")
                continue
            
            # Generate diff
            diff = generate_diff(original_code, final_code)
            
            # Add line numbers
            original_code_with_lines = add_line_numbers(original_code)
            final_code_with_lines = add_line_numbers(final_code)
            
            # Build user prompt
            user_prompt = PROMPT_VERIFICATION["user"].format(
                original_code=original_code_with_lines,
                update_snippet=update_snippet,
                final_code=final_code_with_lines,
                diff=diff
            )
            
            # Calculate token count
            user_tokens = len(tokenizer.encode(user_prompt))
            max_tokens = int(1.8 * user_tokens)
            
            request = {
                "custom_id": custom_id,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens
            }
            
            request_tokens = system_tokens + user_tokens
            
            current_batch.append(request)
            current_batch_tokens += request_tokens
            total_input_tokens += request_tokens
            total_max_tokens += max_tokens
        
        # Save batch requests
        if current_batch:
            output_path = Path(output_dir) / language_type
            write_jsonl(current_batch, str(output_path), file_num)
            logger.info(
                f"[{language_type}] Batch {file_num}: "
                f"{len(current_batch)} requests, {current_batch_tokens} tokens"
            )
    
    return file_count, total_input_tokens, total_max_tokens


def process_language(
    lang: str,
    batch_dir: str,
    codes_dir: str,
    output_dir: str,
    compared_dir: str,
    tokenizer: tiktoken.Encoding
) -> dict[str, Any] | None:
    """
    Process data for a single language.
    
    Returns:
        Processing statistics, or None if no data
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing language: {lang}")
    logger.info(f"{'='*60}")
    
    # Check input directory
    lang_batch_dir = os.path.join(batch_dir, lang)
    lang_codes_dir = os.path.join(codes_dir, lang)
    
    if not os.path.exists(lang_batch_dir):
        logger.warning(f"[{lang}] Batch directory not found: {lang_batch_dir}")
        return None
    
    if not os.path.exists(lang_codes_dir):
        logger.warning(f"[{lang}] Code generation directory not found: {lang_codes_dir}")
        return None
    
    batch_number, input_tokens, max_tokens = load_and_pair_batches(
        batch_dir=lang_batch_dir,
        codes_dir=lang_codes_dir,
        output_dir=output_dir,
        compared_dir=compared_dir,
        language_type=lang,
        tokenizer=tokenizer
    )
    
    if batch_number == 0:
        return None
    
    return {
        "batch_number": batch_number,
        "input_tokens": input_tokens,
        "max_tokens": max_tokens,
    }


def discover_languages(batch_dir: str, codes_dir: str) -> list[str]:
    """Discover all languages that can be processed (must exist in both directories)."""
    batch_langs = set()
    codes_langs = set()
    
    if os.path.exists(batch_dir):
        batch_langs = {
            d for d in os.listdir(batch_dir)
            if os.path.isdir(os.path.join(batch_dir, d))
        }
    
    if os.path.exists(codes_dir):
        codes_langs = {
            d for d in os.listdir(codes_dir)
            if os.path.isdir(os.path.join(codes_dir, d))
        }
    
    # Intersection: must have input and code generation results
    common_langs = batch_langs & codes_langs
    
    return sorted(list(common_langs))


# ============== Result Output ==============

def print_summary(all_stats: dict[str, Any], totals: dict[str, Any]) -> None:
    """Print processing summary information."""
    print("\n" + "=" * 80)
    print("Processing summary")
    print("=" * 80)
    
    print(f"\n{'Language':<20} {'Batches':>10} {'Input Tokens':>15} {'Max Tokens':>15}")
    print("-" * 60)
    
    for lang, stats in all_stats.items():
        print(
            f"{lang:<20} {stats['batch_number']:>10} "
            f"{stats['input_tokens']:>15,} {stats['max_tokens']:>15,}"
        )
    
    print("-" * 60)
    print(
        f"{'Total':<20} {totals['batch_number']:>10} "
        f"{totals['input_tokens']:>15,} {totals['max_tokens']:>15,}"
    )
    
    print(f"\nEstimated cost: ${totals['estimated_cost']:.2f}")


def save_stats(stats_file: str | Path, all_stats: dict, totals: dict) -> None:
    """Save statistics to JSON file."""
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(
            {"languages": all_stats, "grand_total": totals},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nStatistics saved to: {stats_file}")


# ============== Main Function ==============

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-language batch verification data preparation script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--batch_dir",
        default="dataset/synthetic_data/description_combined",
        help="Original code batch directory (output of generate_code.py)",
    )
    parser.add_argument(
        "-c", "--codes_dir",
        default="dataset/synthetic_data/codes",
        help="Code generation result directory (output of send_request_claude_stream.py)",
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="dataset/synthetic_data/codes_processed",
        help="Verification request output directory",
    )
    parser.add_argument(
        "--compared_dir",
        default="dataset/synthetic_data/compared",
        help="Compared file directory",
    )
    parser.add_argument(
        "--tokenizer",
        default="o200k_base",
        help="Tokenizer name",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Specify the list of languages to process (default: all discovered languages)",
    )
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding(args.tokenizer)
    
    # Discover languages to process
    if args.languages:
        languages = args.languages
    else:
        languages = discover_languages(args.batch_dir, args.codes_dir)
    
    if not languages:
        logger.error("No languages found to process, please check input directories")
        return
    
    logger.info(f"Will process the following languages: {languages}")
    
    # Process each language
    all_stats: dict[str, Any] = {}
    totals = {
        "batch_number": 0,
        "input_tokens": 0,
        "max_tokens": 0,
    }
    
    for lang in tqdm(languages, desc="Processing languages", unit="lang"):
        result = process_language(
            lang=lang,
            batch_dir=args.batch_dir,
            codes_dir=args.codes_dir,
            output_dir=args.output_dir,
            compared_dir=args.compared_dir,
            tokenizer=tokenizer,
        )
        
        if result is None:
            continue
        
        all_stats[lang] = result
        totals["batch_number"] += result["batch_number"]
        totals["input_tokens"] += result["input_tokens"]
        totals["max_tokens"] += result["max_tokens"]
    
    # Calculate cost and output results
    totals["estimated_cost"] = calculate_cost(totals["input_tokens"], totals["max_tokens"])
    
    print_summary(all_stats, totals)
    save_stats(Path(args.output_dir) / "verification_stats.json", all_stats, totals)


if __name__ == "__main__":
    main()

