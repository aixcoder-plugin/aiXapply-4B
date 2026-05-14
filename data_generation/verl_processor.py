"""
aiXapply data generation pipeline.

Stage: veRL dataset conversion.
Purpose: build train and test parquet datasets for veRL training.
Inputs: filtered parquet datasets and sample_config_split.json.
Outputs: train.parquet, test.parquet, and dataset statistics.
"""
from __future__ import annotations

import os
import glob
import argparse
import json
import logging
from typing import Any
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from prompt import PROMPT_TRAIN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_split_config(config_path: str) -> dict[str, dict[str, int]]:
    """
    Load split configuration, flattening multi-level categories into per-language split counts.

    Args:
        config_path: Path to configuration file

    Returns:
        Flattened configuration dictionary in {language: {"train_count": n, "test_count": m}} format
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Flatten config: merge languages under different categories
    flat_config = {}
    for category in config.values():
        if isinstance(category, dict):
            for lang, split_info in category.items():
                if isinstance(split_info, dict):
                    train_count = int(split_info.get("train", 0))
                    test_count = int(split_info.get("test", 0))
                else:
                    train_count = 0
                    test_count = int(split_info)

                flat_config[lang.lower()] = {
                    "train_count": train_count,
                    "test_count": test_count,
                }
    return flat_config


def process_judge_files_to_datasets(
    judge_files: list[str],
    compare_dir: str,
    split_config: dict[str, dict[str, int]],
    random_state: int = 42
) -> tuple[Dataset, Dataset, dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
    """
    Process judge result files to generate training and test datasets

    Args:
        judge_files: List of paths to judge result files
        compare_dir: Comparison file directory
        split_config: Split configuration, format {language: {"train": n, "test": m}}
        random_state: Random seed, default 42

    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
        passed_count: Count of passed samples per language
        failed_count: Count of failed samples per language
        split_stats: Split statistics per language {language: {"train": n, "test": m}}
    """
    train_results = []
    test_results = []
    # Count of samples for each language
    passed_count: dict[str, int] = {}
    failed_count: dict[str, int] = {}
    split_stats: dict[str, dict[str, int]] = {}

    for file in tqdm(judge_files, desc="Processing language types", unit="language type"):
        # Extract language type from filename (e.g., python_filtered.parquet -> python)
        filename = Path(file).stem  # python_filtered
        language = filename.replace("_filtered", "").lower()
        
        try:
            df = pd.read_parquet(file)
        except Exception as e:
            logger.error(f"[{language}] Unable to read file {file}: {e}")
            continue

        compare_file_dir = os.path.join(compare_dir, language)

        passed_df = df[df["judge_result"] == "PASSED"].copy()
        failed_df = df[df["judge_result"] == "FAILED"]
        
        # Filter out samples where update_snippet contains "403 - Forbidden"
        forbidden_mask = passed_df["update_snippet"].str.contains("403 - Forbidden", na=False)
        forbidden_count = forbidden_mask.sum()
        if forbidden_count > 0:
            logger.warning(f"[{language}] Filtered out {forbidden_count} samples containing '403 - Forbidden'")
            passed_df = passed_df[~forbidden_mask].copy()

        # Save judge results to compared directory
        for _, d in df.iterrows():
            try:
                task_id = d["custom_id"].split("-")[-1]
                judge_result = d["judge_result"]
                
                output_path = Path(compare_file_dir) / task_id
                output_path.mkdir(parents=True, exist_ok=True)
                
                (output_path / judge_result).write_text(
                    d["judge_content"],
                    encoding="utf-8"
                )
            except Exception as e:
                logger.warning(f"[{language}] Failed to save judge result: {e}")
                continue

        # Statistics
        if language not in passed_count:
            passed_count[language] = 0
        if language not in failed_count:
            failed_count[language] = 0
        passed_count[language] += len(passed_df)
        failed_count[language] += len(failed_df)

        # Split training and test sets by configuration
        if language not in split_config:
            logger.warning(f"[{language}] Split configuration not found in config file, skipping")
            continue

        config = split_config[language]
        test_size = config["test_count"]
        
        # Randomly shuffle data for this language
        passed_df = passed_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split by configured test count
        actual_test_size = min(test_size, len(passed_df))
        if actual_test_size < test_size:
            logger.warning(f"[{language}] Insufficient data, adjusting test set size from {test_size} to {actual_test_size}")
        
        if actual_test_size > 0:
            test_df = passed_df.iloc[:actual_test_size].copy()
            train_df = passed_df.iloc[actual_test_size:].copy()
        else:
            train_df = passed_df.copy()
            test_df = pd.DataFrame(columns=passed_df.columns)

        # Add language type to data
        train_df["language"] = language
        test_df["language"] = language
        
        train_results.append(train_df)
        test_results.append(test_df)
        
        # Record split statistics
        split_stats[language] = {
            "train": len(train_df),
            "test": len(test_df)
        }

    if not train_results and not test_results:
        logger.error("No passed samples found")
        # Return empty datasets
        empty_df = pd.DataFrame(columns=[
            "custom_id", "original_code", "update_snippet", 
            "final_code", "judge_result", "judge_content", "language"
        ])
        return (
            Dataset.from_pandas(empty_df, split="train"),
            Dataset.from_pandas(empty_df, split="test"),
            passed_count,
            failed_count,
            split_stats
        )

    # Merge all results
    train_df = pd.concat(train_results, ignore_index=True) if train_results else pd.DataFrame()
    test_df = pd.concat(test_results, ignore_index=True) if test_results else pd.DataFrame()
    
    # Drop unnecessary columns
    cols_to_drop = ["judge_content", "custom_id", "judge_result"]
    for col in cols_to_drop:
        if col in train_df.columns:
            train_df.drop(columns=[col], inplace=True)
        if col in test_df.columns:
            test_df.drop(columns=[col], inplace=True)

    # Randomly shuffle final datasets
    if len(train_df) > 0:
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    if len(test_df) > 0:
        test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Load using datasets library
    train_dataset = Dataset.from_pandas(train_df, split="train")
    test_dataset = Dataset.from_pandas(test_df, split="test")

    return train_dataset, test_dataset, passed_count, failed_count, split_stats


def make_map_fn(split: str):
    """
    Create data mapping function

    Args:
        split: Dataset split type ("train" or "test")

    Returns:
        Processing function
    """
    def process_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        original_code = example.pop("original_code")
        update_snippet = example.pop("update_snippet")
        final_code = example.pop("final_code")
        language = example.pop("language")

        data = {
            "data_source": "verl/aiXapply",
            "prompt": [
                {
                    "role": "system",
                    "content": PROMPT_TRAIN["system"],
                },
                {
                    "role": "user",
                    "content": PROMPT_TRAIN["user"].format(
                        language=language,
                        source_file=original_code,
                        update_snippet=update_snippet
                    ),
                }
            ],
            "reward_model": {"style": "rule", "ground_truth": final_code},
            "extra_info": {
                "split": split,
                "index": idx,
                "original_code": original_code,
                "update_snippet": update_snippet,
                "language": language,
            },
        }
        return data

    return process_fn


def discover_judge_files(judge_dir: str) -> list[str]:
    """
    Discover all judge result files

    Args:
        judge_dir: Judge results directory

    Returns:
        List of judge file paths
    """
    judge_files = glob.glob(os.path.join(judge_dir, "*_filtered.parquet"))
    return sorted(judge_files)


def merge_datasets(
    dataset1: Dataset | None,
    dataset2: Dataset | None,
    split: str,
    random_state: int = 42
) -> Dataset:
    """
    Merge two datasets

    Args:
        dataset1: Dataset 1
        dataset2: Dataset 2
        split: Dataset split type
        random_state: Random seed

    Returns:
        Merged dataset
    """
    from datasets import concatenate_datasets
    
    datasets_to_merge = []
    if dataset1 is not None and len(dataset1) > 0:
        datasets_to_merge.append(dataset1)
    if dataset2 is not None and len(dataset2) > 0:
        datasets_to_merge.append(dataset2)
    
    if not datasets_to_merge:
        # Return empty dataset
        empty_df = pd.DataFrame(columns=["data_source", "prompt", "reward_model", "extra_info"])
        return Dataset.from_pandas(empty_df, split=split)
    
    if len(datasets_to_merge) == 1:
        merged = datasets_to_merge[0]
    else:
        merged = concatenate_datasets(datasets_to_merge)
    
    # Randomly shuffle
    merged = merged.shuffle(seed=random_state)
    
    return merged


def print_summary(
    passed_count: dict[str, int],
    failed_count: dict[str, int],
    split_stats: dict[str, dict[str, int]],
    train_len: int,
    test_len: int
) -> None:
    """Print processing summary information"""
    print("\n" + "=" * 80)
    print("Processing Summary")
    print("=" * 80)

    print(f"\n{'Language':<20} {'Passed':>10} {'Failed':>10} {'Pass Rate':>12} {'Train':>10} {'Test':>10}")
    print("-" * 80)

    total_passed = 0
    total_failed = 0
    total_train = 0
    total_test = 0

    # Sort output by language
    all_languages = set(passed_count.keys()) | set(failed_count.keys())
    for lang in sorted(all_languages):
        passed = passed_count.get(lang, 0)
        failed = failed_count.get(lang, 0)
        total = passed + failed
        rate = f"{passed / total * 100:.1f}%" if total > 0 else "N/A"
        
        train_cnt = split_stats.get(lang, {}).get("train", 0)
        test_cnt = split_stats.get(lang, {}).get("test", 0)
        
        print(f"{lang:<20} {passed:>10} {failed:>10} {rate:>12} {train_cnt:>10} {test_cnt:>10}")
        total_passed += passed
        total_failed += failed
        total_train += train_cnt
        total_test += test_cnt

    print("-" * 80)
    total = total_passed + total_failed
    total_rate = f"{total_passed / total * 100:.1f}%" if total > 0 else "N/A"
    print(f"{'Total':<20} {total_passed:>10} {total_failed:>10} {total_rate:>12} {total_train:>10} {total_test:>10}")

    print(f"\nDataset Statistics:")
    print(f" Training set: {train_len:,} items")
    print(f" Test set: {test_len:,} items")


def save_stats(
    stats_file: str,
    passed_count: dict[str, int],
    failed_count: dict[str, int],
    split_stats: dict[str, dict[str, int]],
    train_len: int,
    test_len: int
) -> None:
    """Save statistics to JSON file"""
    stats = {
        "languages": {
            lang: {
                "passed": passed_count.get(lang, 0),
                "failed": failed_count.get(lang, 0),
                "train": split_stats.get(lang, {}).get("train", 0),
                "test": split_stats.get(lang, {}).get("test", 0)
            }
            for lang in set(passed_count.keys()) | set(failed_count.keys())
        },
        "dataset": {
            "train_size": train_len,
            "test_size": test_len
        }
    }
    
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Statistics saved to: {stats_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-language VERL data processing script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--judge_dir",
        default="dataset/synthetic_data/filtered",
        type=str,
        help="Directory containing judge results"
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="dataset/synthetic_data/final_dataset",
        type=str,
        help="Output directory"
    )
    parser.add_argument(
        "-d", "--compare_dir",
        default="dataset/synthetic_data/compared",
        type=str,
        help="Comparison file directory"
    )
    parser.add_argument(
        "-c", "--split_config",
        default="data_generation/sample_config_split.json",
        type=str,
        help="Split configuration file path, specifies train/test count for each language"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = parse_args()

    # Load split configuration
    if not os.path.exists(args.split_config):
        logger.error(f"Split configuration file does not exist: {args.split_config}")
        return
    
    split_config = load_split_config(args.split_config)
    logger.info(f"Loaded split configuration, total {len(split_config)} languages")

    # Discover judge result files
    judge_files = discover_judge_files(args.judge_dir)

    if not judge_files:
        logger.error(f"Judge result files not found, please check directory: {args.judge_dir}")
        logger.info("Expected file format: {judge_dir}/{language}_filtered.parquet")
        return

    logger.info(f"Found {len(judge_files)} judge result files")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process data
    train_dataset, test_dataset, passed_count, failed_count, split_stats = process_judge_files_to_datasets(
        judge_files=judge_files,
        compare_dir=args.compare_dir,
        split_config=split_config,
        random_state=args.random_state
    )

    # Print summary
    print_summary(passed_count, failed_count, split_stats, len(train_dataset), len(test_dataset))

    # Apply mapping function
    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        desc="Processing train dataset"
    )
    
    test_dataset = test_dataset.map(
        function=make_map_fn("test"),
        with_indices=True,
        desc="Processing test dataset"
    )

    # Save datasets
    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    
    train_dataset = train_dataset.shuffle(seed=args.random_state)
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    
    logger.info(f"Training set saved to: {train_path}")
    logger.info(f"Test set saved to: {test_path}")

    # Save statistics
    stats_file = os.path.join(args.output_dir, "stats.json")
    final_train_len = len(train_dataset)
    final_test_len = len(test_dataset)
    save_stats(stats_file, passed_count, failed_count, split_stats, final_train_len, final_test_len)

if __name__ == "__main__":
    main()