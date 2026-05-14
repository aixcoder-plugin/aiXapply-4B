"""
aiXapply data generation pipeline.

Stage: source sample preparation.
Purpose: filter CommitPack samples and build description-generation batches.
Inputs: CommitPack sample files and sample_config.json.
Outputs: batch_*.jsonl files and comparison artifacts.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken
from tqdm import tqdm

from prompt import PROMPT_DESCRIPTION
from utils import EXTENSION_MAP, calculate_cost, write_jsonl

# ============== Configuration Constants ==============

# Data filtering parameters
CHAR_LENGTH_RANGE = (45, 70_000)       # Character length range
LINE_COUNT_RANGE = (20, 1_400)         # Line count range
LINE_RATIO_RANGE = (0.7, 1.3)          # New/old code line ratio range
MIN_SUBJECT_LENGTH = 10                # Minimum commit message length
MIN_LINE_COUNT = 50                    # Minimum code line count

# Model parameters
DEFAULT_MAX_TOKENS = 8192
DEFAULT_BATCH_LIMIT = 90000  # Default batch token limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============== Utility Functions ==============

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load sampling configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load data from file into DataFrame.
    
    Supported formats: .jsonl, .parquet, .csv
    """
    file_path = str(file_path)
    
    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            return pd.DataFrame([json.loads(line) for line in f if line.strip()])
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def get_file_extension(lang: str) -> str:
    """Get file extension based on language type."""
    return EXTENSION_MAP.get(lang.lower(), lang.lower())


# ============== Data Processing ==============

def data_filter(df: pd.DataFrame, lang: str | None = None) -> pd.DataFrame:
    """
    Filter data, keeping only rows that meet the criteria.
    
    Filtering criteria:
    - old_file == new_file
    - old_contents, new_contents, subject are not null
    - Character count and line count are within specified ranges
    - Ratio of new to old code lines is within specified range
    - subject length >= 10 and code line count > 50
    """
    tag = f"[{lang}]" if lang else ""
    logger.info(f"{tag} Data filtering in progress...")
    logger.info(f"{tag} Before filtering: {df.shape}")
    
    # Basic filtering
    mask = (
        (df["old_file"] == df["new_file"]) &
        df["old_contents"].notna() &
        df["new_contents"].notna() &
        df["subject"].notna()
    )
    df = df[mask].copy()
    
    if df.empty:
        logger.info(f"{tag} After filtering: (0, 0)")
        return df
    
    # Calculate length metrics
    old_char_len = df["old_contents"].str.len()
    new_char_len = df["new_contents"].str.len()
    old_line_count = df["old_contents"].str.split("\n").str.len()
    new_line_count = df["new_contents"].str.split("\n").str.len()
    
    # Character count filtering
    char_mask = (
        old_char_len.between(*CHAR_LENGTH_RANGE) &
        new_char_len.between(*CHAR_LENGTH_RANGE)
    )
    
    # Line count filtering
    line_mask = (
        old_line_count.between(*LINE_COUNT_RANGE) &
        new_line_count.between(*LINE_COUNT_RANGE)
    )
    
    # Line ratio filtering
    line_ratio = old_line_count / new_line_count
    ratio_mask = line_ratio.between(*LINE_RATIO_RANGE)
    
    # subject length and code line count filtering
    quality_mask = (
        (df["subject"].str.len() >= MIN_SUBJECT_LENGTH) &
        (old_line_count > MIN_LINE_COUNT)
    )

    # Remove samples where old_contents is entirely numeric (only digits after removing whitespace)
    numeric_only_mask = ~df["old_contents"].str.replace(r'\s', '', regex=True).str.fullmatch(r'\d+')

    df = df[char_mask & line_mask & ratio_mask & quality_mask & numeric_only_mask].reset_index(drop=True)
    logger.info(f"{tag} After filtering: {df.shape}")
    
    return df


def sample_data(df: pd.DataFrame, sample_count: int, seed: int = 42) -> pd.DataFrame:
    """
    Sample a specified number of data points from the DataFrame.
    
    If available data is insufficient, all available data will be used.
    """
    available = len(df)
    
    if available == 0:
        return pd.DataFrame()
    
    if available < sample_count:
        logger.warning(
            f"Available samples ({available}) < Required samples ({sample_count}), using all available data"
        )
        actual_count = available
    else:
        actual_count = sample_count
    
    # Shuffle data and sample
    return df.sample(n=actual_count, random_state=seed).reset_index(drop=True)


# ============== File Search and Output ==============

def find_language_files(input_dir: str | Path, lang: str) -> list[str]:
    """
    Find data files for the specified language.
    
    Will try multiple possible directory name formats (original name, case variants, special character replacements).
    """
    possible_names = {
        lang,
        lang.lower(),
        lang.upper(),
        lang.replace("+", "plus"),
        lang.replace("#", "sharp"),
    }
    
    files: set[str] = set()
    for name in possible_names:
        for ext in ("jsonl", "parquet", "csv"):
            pattern = os.path.join(input_dir, name, f"**/*.{ext}")
            files.update(glob.glob(pattern, recursive=True))
    
    return list(files)


def save_comparison_files(
    compare_dir: str | Path,
    lang: str,
    index: int,
    old_contents: str,
    new_contents: str,
    commit_message: str,
) -> None:
    """Save comparison files (original code, modified code, commit message)."""
    ext = get_file_extension(lang)
    output_path = Path(compare_dir) / lang / str(index)
    output_path.mkdir(parents=True, exist_ok=True)
    
    (output_path / f"old_contents.{ext}").write_text(old_contents, encoding="utf-8")
    (output_path / f"new_contents.{ext}").write_text(new_contents, encoding="utf-8")
    (output_path / "commit_message.txt").write_text(commit_message, encoding="utf-8")


def save_batch_requests(
    requests: list[dict],
    output_dir: str | Path,
    lang: str,
    batch_number: int = 1,
) -> None:
    """Save batch requests to JSONL file."""
    if not requests:
        return
 
    output_path = Path(output_dir) / lang
    write_jsonl(requests, str(output_path), batch_number)
    logger.info(f"[{lang}] Saved {len(requests)} requests to batch_{batch_number:03d}.jsonl")


# ============== Core Processing Logic ==============

def prepare_batch_requests(
    df: pd.DataFrame,
    lang: str,
    tokenizer: tiktoken.Encoding,
    output_dir: str,
    compare_dir: str,
    batch_limit: int = DEFAULT_BATCH_LIMIT,
) -> tuple[int, int, int]:
    """
    Prepare batch requests and save comparison files.
    
    When batch token count exceeds batch_limit, it will automatically split into multiple batch files.
    
    Returns:
        (Request count, total input tokens, total max tokens)
    """
    batch_number = 1
    current_batch: list[dict] = []
    current_batch_tokens = 0
    total_input_tokens = 0
    total_max_tokens = 0
    total_requests = 0
    
    for index, row in tqdm(df.iterrows(), desc=f"Preparing batch requests for {lang}"):
        commit_message = row["subject"]
        if "403 - Forbidden" in commit_message:  # skip the data that is forbidden to access
            continue
        old_contents = row["old_contents"]
        new_contents = row["new_contents"]
        
        # Build prompt
        user_prompt = PROMPT_DESCRIPTION["user"].format(
            language=lang,
            original_code=old_contents,
            changed_code=new_contents,
            commit_message=commit_message,
        )
        
        # Calculate token count
        messages = [
            {"role": "system", "content": PROMPT_DESCRIPTION["system"]},
            {"role": "user", "content": user_prompt},
        ]
        request_tokens = sum(
            len(tokenizer.encode(msg["content"])) for msg in messages
        )
        
        # Create request
        request = {
            "custom_id": f"{lang}-{index}",
            "user_prompt": user_prompt,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        
        # Save comparison files
        save_comparison_files(
            compare_dir, lang, index,
            old_contents, new_contents, commit_message,
        )
        
        # Check if batch split is needed
        if current_batch_tokens + request_tokens > batch_limit:
            save_batch_requests(current_batch, output_dir, lang, batch_number)
            print(f"Batch {batch_number}: {len(current_batch)} requests, {current_batch_tokens} tokens")
            batch_number += 1
            current_batch = []
            current_batch_tokens = 0
        
        current_batch.append(request)
        current_batch_tokens += request_tokens
        total_input_tokens += request_tokens
        total_max_tokens += DEFAULT_MAX_TOKENS
        total_requests += 1
    
    # Save last batch
    if current_batch:
        save_batch_requests(current_batch, output_dir, lang, batch_number)
        print(f"Batch {batch_number}: {len(current_batch)} requests, {current_batch_tokens} tokens")
    
    return total_requests, total_input_tokens, total_max_tokens


def process_language(
    lang: str,
    sample_count: int,
    input_dir: str,
    output_dir: str,
    compare_dir: str,
    tokenizer: tiktoken.Encoding,
    seed: int,
    batch_limit: int = DEFAULT_BATCH_LIMIT,
) -> dict[str, Any] | None:
    """
    Process data for a single language.
    
    Returns:
        Language processing statistics, or None if no data
    """
    logger.info(f"\n[{lang}] Target sample count: {sample_count}")
    
    # Find data files
    files = find_language_files(input_dir, lang)
    if not files:
        logger.warning(f"[{lang}] No data files found, skipping...")
        return None
    
    logger.info(f"[{lang}] Found {len(files)} data files")
    
    # Load and merge data
    dfs: list[pd.DataFrame] = []
    for file in files:
        try:
            dfs.append(load_data(file))
        except Exception as e:
            logger.error(f"[{lang}] Failed to load {file}: {e}")
    
    if not dfs:
        logger.warning(f"[{lang}] No valid data, skipping...")
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"[{lang}] Merged data: {combined_df.shape}")
    
    # Filter data
    filtered_df = data_filter(combined_df, lang)
    if filtered_df.empty:
        logger.warning(f"[{lang}] No data after filtering, skipping...")
        return None
    
    # Sample data
    sampled_df = sample_data(filtered_df, sample_count, seed)
    logger.info(f"[{lang}] Sampling result: {len(sampled_df)} items")
    
    # Prepare batch requests
    if sampled_df.empty:
        return None
    
    count, input_tokens, max_tokens = prepare_batch_requests(
        sampled_df, lang, tokenizer, output_dir, compare_dir, batch_limit
    )
    
    return {
        "count": count,
        "input_tokens": input_tokens,
        "max_tokens": max_tokens,
    }


def process_category(
    category: str,
    languages: dict[str, int],
    input_dir: str,
    output_dir: str,
    compare_dir: str,
    tokenizer: tiktoken.Encoding,
    seed: int = 42,
    batch_limit: int = DEFAULT_BATCH_LIMIT,
) -> dict[str, Any]:
    """Process all languages under a category."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing category: {category}")
    logger.info(f"{'='*60}")
    
    stats = {
        "category": category,
        "languages": {},
        "total_count": 0,
        "total_input_tokens": 0,
        "total_max_tokens": 0,
    }
    
    for lang, sample_count in tqdm(languages.items(), desc=f"Processing {category}"):
        result = process_language(
            lang=lang,
            sample_count=sample_count,
            input_dir=input_dir,
            output_dir=output_dir,
            compare_dir=compare_dir,
            tokenizer=tokenizer,
            seed=seed,
            batch_limit=batch_limit,
        )
        
        if result is None:
            continue
        
        stats["languages"][lang] = result["count"]
        stats["total_count"] += result["count"]
        stats["total_input_tokens"] += result["input_tokens"]
        stats["total_max_tokens"] += result["max_tokens"]
    
    return stats


# ============== Result Output ==============

def print_summary(all_stats: list[dict[str, Any]], totals: dict[str, Any]) -> None:
    """Print processing summary."""
    print("\n" + "=" * 80)
    print("Processing Summary")
    print("=" * 80)
    
    for stats in all_stats:
        print(f"\n{stats['category']}")
        print(f" {'Language':<20} {'Count':>10}")
        print(f" {'-'*30}")
        for lang, count in stats["languages"].items():
            print(f" {lang:<20} {count:>10}")
        print(f" {'-'*30}")
        print(f" {'Subtotal':<20} {stats['total_count']:>10}")
    
    print(f"\n{'='*80}")
    print("Grand Total")
    print("=" * 80)
    print(f"Total samples:    {totals['count']:,}")
    print(f"\nInput tokens: {totals['input_tokens']:,}")
    print(f"Max tokens: {totals['max_tokens']:,}")
    print(f"\nEstimated cost: ${totals['estimated_cost']:.2f}")


def save_stats(stats_file: str | Path, all_stats: list, totals: dict) -> None:
    """Save statistics to JSON file."""
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(
            {"categories": all_stats, "grand_total": totals},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nStatistics saved to: {stats_file}")


# ============== Main Function ==============

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-language dataset construction script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        default="data_generation/sample_config.json",
        help="Sampling configuration file path",
    )
    parser.add_argument(
        "-i", "--input_dir",
        default="dataset/commitpack_samples/data",
        help="Input data directory",
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="dataset/synthetic_data/batch_data",
        help="Output directory",
    )
    parser.add_argument(
        "-d", "--compare_dir",
        default="dataset/synthetic_data/compared",
        help="Comparison file directory",
    )
    parser.add_argument(
        "--tokenizer",
        default="o200k_base",
        help="Tokenizer name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--batch_limit",
        type=int,
        default=DEFAULT_BATCH_LIMIT,
        help="Maximum tokens per batch",
    )
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration: {args.config}")
    config = load_config(args.config)
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding(args.tokenizer)
    
    # Determine categories to process
    categories = list(config.keys())
    
    # Process each category
    all_stats: list[dict[str, Any]] = []
    totals = {
        "count": 0,
        "input_tokens": 0,
        "max_tokens": 0,
    }
    
    for category in tqdm(categories, desc="Processing categories"):
        stats = process_category(
            category=category,
            languages=config[category],
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            compare_dir=args.compare_dir,
            tokenizer=tokenizer,
            seed=args.seed,
            batch_limit=args.batch_limit,
        )
        
        all_stats.append(stats)
        totals["count"] += stats["total_count"]
        totals["input_tokens"] += stats["total_input_tokens"]
        totals["max_tokens"] += stats["total_max_tokens"]
    
    # Calculate cost and output results
    totals["estimated_cost"] = calculate_cost(totals["input_tokens"], totals["max_tokens"])
    
    print_summary(all_stats, totals)
    save_stats(Path(args.output_dir) / "processing_stats.json", all_stats, totals)


if __name__ == "__main__":
    main()
