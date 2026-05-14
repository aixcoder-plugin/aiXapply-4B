"""
aiXapply data generation pipeline.

Stage: code generation batch preparation.
Purpose: pair original samples with descriptions and build Claude requests.
Inputs: batch_*.jsonl files and description output_batch_*.jsonl files.
Outputs: code generation batch_*.jsonl files and comparison artifacts.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any

import tiktoken
from tqdm import tqdm

from prompt import PROMPT_CODE_GENERATION
from utils import calculate_cost, update_data_pair, write_jsonl

# ============== Configuration Constants ==============

# Language extension mapping
EXTENSION_MAP: dict[str, str] = {
    # Programming languages
    "python": "py",
    "javascript": "js",
    "java": "java",
    "c++": "cpp",
    "go": "go",
    "typescript": "ts",
    "c#": "cs",
    "rust": "rs",
    # Configuration files
    "json": "json",
    "yaml": "yaml",
    "xml": "xml",
    "sql": "sql",
    "ini": "ini",
    # Documentation
    "markdown": "md",
    "text": "txt",
    "html": "html",
    "restructuredtext": "rst",
    # DevOps
    "shell": "sh",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============== Utility Functions ==============

def get_file_extension(lang: str) -> str:
    """Get file extension based on language type."""
    return EXTENSION_MAP.get(lang.lower(), lang.lower())


def extract_change_description(content: str) -> str:
    """Extract change_description from response content."""
    try:
        return content.split("<change_description>")[1].split("</change_description>")[0].strip()
    except (IndexError, AttributeError):
        return content


def save_change_description(
    compared_dir: str | Path,
    lang: str,
    task_num: str,
    change_description: str
) -> None:
    """Save change_description to the comparison directory."""
    output_path = Path(compared_dir) / lang / str(task_num)
    output_path.mkdir(parents=True, exist_ok=True)
    
    (output_path / "change_description.txt").write_text(
        change_description,
        encoding="utf-8"
    )


# ============== Core Processing Logic ==============

def load_and_pair_batches(
    batch_dir: str,
    describe_dir: str,
    output_dir: str,
    compared_dir: str,
    language_type: str,
    tokenizer: tiktoken.Encoding
) -> tuple[int, int, int]:
    """
    Load batch files and pair them with description results.
    
    Args:
        batch_dir: Original code batch directory (contains input requests)
        describe_dir: Description generation results directory (contains output results)
        output_dir: Code generation request output directory
        compared_dir: Comparison files directory
        language_type: Language type
        tokenizer: Tokenizer
    
    Returns:
        (batch_count, total_input_tokens, total_max_tokens)
    """
    # Find input batch files
    input_files = sorted(glob.glob(os.path.join(batch_dir, "batch_*.jsonl")))
    
    if not input_files:
        logger.warning(f"[{language_type}] No input batch files found: {batch_dir}")
        return 0, 0, 0
    
    total_input_tokens = 0
    total_max_tokens = 0
    system_prompt = PROMPT_CODE_GENERATION["system"]
    system_tokens = len(tokenizer.encode(system_prompt))
    
    for input_file in tqdm(
        input_files, 
        # input_files[:len(input_files)//2], # ATTENTION: only process half of the files to save cost
        desc=f"[{language_type}] Processing batch files",
        unit="file"
    ):
        data_pair = {}
        current_batch = []
        current_batch_tokens = 0
        
        # Parse batch number
        file_num = int(os.path.basename(input_file).split("_")[-1].split(".")[0])
        describe_file = os.path.join(describe_dir, f"output_batch_{file_num:03d}.jsonl")
        
        # Check if description file exists
        if not os.path.exists(describe_file):
            logger.warning(f"[{language_type}] Description file not found: {describe_file}")
            continue
        
        # Read input and description files separately and join by custom_id.
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line_in in f_in:
                data_in = json.loads(line_in)
                user_prompt_in = data_in['user_prompt']
                custom_id = data_in['custom_id']
                
                # Extract original code and commit message
                try:
                    original_code = user_prompt_in.split("<original_code>")[1].split("</original_code>")[0].strip()
                    commit_message = user_prompt_in.split("<commit_message>")[1].split("</commit_message>")[0].strip()
                except (IndexError, AttributeError):
                    logger.error(f"[{language_type}] Parsing failed: {custom_id}")
                    continue
                
                data_pair = update_data_pair(data_pair, custom_id, {
                    'original_code': original_code,
                    'commit_message': commit_message,
                    'request': data_in
                })

        with open(describe_file, 'r', encoding='utf-8') as f_des:
            for line_des in f_des:
                data_out = json.loads(line_des)
                custom_id_out = data_out['custom_id']
                
                try:
                    change_description_warp = data_out['response']['body']['choices'][0]['message']['content']
                except (KeyError, IndexError):
                    logger.error(f"[{language_type}] Failed to parse description response: {custom_id_out}")
                    continue
                
                change_description = extract_change_description(change_description_warp)
                
                # Save change_description to comparison directory
                task_num = custom_id_out.split("-")[-1]
                save_change_description(compared_dir, language_type, task_num, change_description)
                
                data_pair = update_data_pair(data_pair, custom_id_out, {
                    'change_description': change_description
                })
        
        # Step 2: Generate code generation requests
        for custom_id, data in tqdm(
            data_pair.items(),
            desc=f"[{language_type}] Generating requests",
            unit="pair",
            leave=False
        ):
            original_code = data.get('original_code')
            commit_message = data.get('commit_message')
            change_description = data.get('change_description')
            
            if not all([original_code, commit_message, change_description]):
                logger.warning(f"[{language_type}] Incomplete data: {custom_id}")
                continue
            
            # Build user prompt (multi-language version includes language parameter)
            user_prompt = PROMPT_CODE_GENERATION["user"].format(
                language=language_type,
                original_code=original_code,
                brief_change=commit_message,
                description=change_description
            )
            
            # Calculate token count
            user_tokens = len(tokenizer.encode(user_prompt))
            max_tokens = int(10 * user_tokens)  # text file size is usually larger than code file size.
            
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
    
    return len(input_files), total_input_tokens, total_max_tokens


def process_language(
    lang: str,
    batch_dir: str,
    describe_dir: str,
    output_dir: str,
    compared_dir: str,
    tokenizer: tiktoken.Encoding
) -> dict[str, Any] | None:
    """
    Process data for a single language.
    
    Returns:
        Processing statistics, or None if no data available
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing language: {lang}")
    logger.info(f"{'='*60}")
    
    # Check input directories
    lang_batch_dir = os.path.join(batch_dir, lang)
    lang_describe_dir = os.path.join(describe_dir, lang)
    
    if not os.path.exists(lang_batch_dir):
        logger.warning(f"[{lang}] Batch directory not found: {lang_batch_dir}")
        return None
    
    if not os.path.exists(lang_describe_dir):
        logger.warning(f"[{lang}] Description directory not found: {lang_describe_dir}")
        return None
    
    batch_number, input_tokens, max_tokens = load_and_pair_batches(
        batch_dir=lang_batch_dir,
        describe_dir=lang_describe_dir,
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


def discover_languages(batch_dir: str, describe_dir: str) -> list[str]:
    """Discover all processable languages (must exist in both directories)."""
    batch_langs = set()
    describe_langs = set()
    
    if os.path.exists(batch_dir):
        batch_langs = {
            d for d in os.listdir(batch_dir)
            if os.path.isdir(os.path.join(batch_dir, d))
        }
    
    if os.path.exists(describe_dir):
        describe_langs = {
            d for d in os.listdir(describe_dir)
            if os.path.isdir(os.path.join(describe_dir, d))
        }
    
    # Intersection: must have both input and description results
    common_langs = batch_langs & describe_langs
    
    return sorted(list(common_langs))


# ============== Result Output ==============

def print_summary(all_stats: dict[str, Any], totals: dict[str, Any]) -> None:
    """Print processing summary."""
    print("\n" + "=" * 80)
    print("Processing Summary")
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
    """Save statistics to a JSON file."""
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
        description="Multi-language code generation data preparation script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--batch_dir",
        default="dataset/synthetic_data/batch_data",
        help="Original code batch directory (output of prepare_batch_data.py)",
    )
    parser.add_argument(
        "-d", "--describe_dir",
        default="dataset/synthetic_data/description",
        help="Description generation results directory (output of send_request_openai.py)",
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="dataset/synthetic_data/description_combined",
        help="Code generation request output directory",
    )
    parser.add_argument(
        "-c", "--compared_dir",
        default="dataset/synthetic_data/compared",
        help="Comparison files directory",
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
        help="Specify the list of languages to process (defaults to all discovered languages)",
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
        languages = discover_languages(args.batch_dir, args.describe_dir)
    
    if not languages:
        logger.error("No processable languages found, please check the input directories")
        return
    
    logger.info(f"Languages to process: {languages}")
    
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
            describe_dir=args.describe_dir,
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
    save_stats(Path(args.output_dir) / "code_gen_stats.json", all_stats, totals)


if __name__ == "__main__":
    main()

