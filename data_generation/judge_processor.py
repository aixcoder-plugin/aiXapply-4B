"""
aiXapply data generation pipeline.

Stage: judgment result processing.
Purpose: join original samples, generated code, and LLM judgment results.
Inputs: source batch_*.jsonl, code output_batch_*.jsonl, and judge output_batch_*.jsonl files.
Outputs: per-language judged and passed parquet datasets.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from utils import update_data_pair


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============== Core Processing Logic ==============

def process_judge_results(
    origin_code_dir: str,
    codes_dir: str,
    judge_dir: str,
    language_type: str
) -> list[dict[str, Any]]:
    """
    Process judgment results for a single language.
    
    Args:
        origin_code_dir: Original code batch directory
        codes_dir: Code generation result directory
        judge_dir: Verification result directory
        language_type: Language type
    
    Returns:
        List of processed results
    """
    input_files = sorted(glob.glob(os.path.join(origin_code_dir, "batch_*.jsonl")))
    
    if not input_files:
        logger.warning(f"[{language_type}] No input batch files found: {origin_code_dir}")
        return []
    
    all_results = []
    
    for input_file in tqdm(
        input_files,
        desc=f"[{language_type}] Processing batch files",
        unit="file",
        leave=False
    ):
        # Parse batch number
        file_num = int(os.path.basename(input_file).split("_")[-1].split(".")[0])
        codes_file = os.path.join(codes_dir, f"output_batch_{file_num:03d}.jsonl")
        judge_file = os.path.join(judge_dir, f"output_batch_{file_num:03d}.jsonl")
        
        # Check if files exist
        if not os.path.exists(codes_file):
            logger.debug(f"[{language_type}] Code file not found: {codes_file}")
            continue
        if not os.path.exists(judge_file):
            logger.debug(f"[{language_type}] Verification file not found: {judge_file}")
            continue
        
        data_pair = {}
        
        # Read each source independently and join by custom_id. This avoids
        # silent row shifts when an upstream request fails or completes out of order.
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in, \
                 open(codes_file, 'r', encoding='utf-8') as f_codes, \
                 open(judge_file, 'r', encoding='utf-8') as f_judge:
                
                for line_codes in f_codes:
                    try:
                        data_codes = json.loads(line_codes)
                        custom_id = data_codes['custom_id']
                        raw_content = data_codes['response']['body']['choices'][0]['message']['content']
                        
                        # Extract update_snippet
                        update_snippets = re.findall(
                            r"<update_snippet>(.*?)</update_snippet>",
                            raw_content,
                            flags=re.DOTALL
                        )
                        update_snippet = "".join(update_snippets).strip()
                        
                        # Extract final_code
                        try:
                            final_code = raw_content.split("<final_code>")[1].split("</final_code>")[0].strip()
                        except (IndexError, AttributeError):
                            final_code = "ERROR"
                            logger.debug(f"[{language_type}] Failed to extract final_code: {custom_id}")
                        
                        data_pair = update_data_pair(data_pair, custom_id, {
                            'update_snippet': update_snippet,
                            'final_code': final_code,
                        })
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug(f"[{language_type}] Failed to parse code results: {e}")
                        continue
                    
                for line_in in f_in:
                    try:
                        data_in = json.loads(line_in)
                        custom_id_in = data_in['custom_id']
                        user_prompt = data_in['user_prompt']
                        
                        # Extract original_code
                        original_code = user_prompt.split("<original_code>")[1].split("</original_code>")[0].strip()
                        
                        data_pair = update_data_pair(data_pair, custom_id_in, {
                            'original_code': original_code,
                        })
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.debug(f"[{language_type}] Failed to parse original code: {e}")
                        continue
                    
                for line_judge in f_judge:
                    try:
                        data_judge = json.loads(line_judge)
                        custom_id_judge = data_judge['custom_id']
                        judge_content = data_judge['response']['body']['choices'][0]['message']['content']
                        
                        # Determine judge result
                        judge_result = "PASSED" if "<Judge>PASSED</Judge>" in judge_content else "FAILED"
                        
                        data_pair = update_data_pair(data_pair, custom_id_judge, {
                            'judge_result': judge_result,
                            'judge_content': judge_content
                        })
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug(f"[{language_type}] Failed to parse verification results: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"[{language_type}] Failed to process file: {input_file}: {e}")
            continue
        
        # Collect complete data pairs
        required_keys = ['original_code', 'update_snippet', 'final_code', 'judge_result']
        for custom_id, data in data_pair.items():
            if all(k in data for k in required_keys):
                all_results.append({
                    'custom_id': custom_id,
                    'language': language_type,
                    'original_code': data['original_code'],
                    'update_snippet': data['update_snippet'],
                    'final_code': data['final_code'],
                    'judge_result': data['judge_result'],
                    'judge_content': data.get('judge_content', '')
                })
    
    return all_results


def process_language(
    lang: str,
    batch_dir: str,
    codes_dir: str,
    judge_dir: str,
    output_dir: str
) -> dict[str, Any] | None:
    """
    Process data for a single language.
    
    Returns:
        Processing statistics, or None if no data
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing language: {lang}")
    logger.info(f"{'='*60}")
    
    # Build language-specific directory paths
    lang_batch_dir = os.path.join(batch_dir, lang)
    lang_codes_dir = os.path.join(codes_dir, lang)
    lang_judge_dir = os.path.join(judge_dir, lang)
    
    # Check if directories exist
    if not os.path.exists(lang_batch_dir):
        logger.warning(f"[{lang}] Batch directory not found: {lang_batch_dir}")
        return None
    if not os.path.exists(lang_codes_dir):
        logger.warning(f"[{lang}] Code directory not found: {lang_codes_dir}")
        return None
    if not os.path.exists(lang_judge_dir):
        logger.warning(f"[{lang}] Verification directory not found: {lang_judge_dir}")
        return None
    
    # Process judgment results
    results = process_judge_results(
        origin_code_dir=lang_batch_dir,
        codes_dir=lang_codes_dir,
        judge_dir=lang_judge_dir,
        language_type=lang
    )
    
    if not results:
        logger.warning(f"[{lang}] No valid results found")
        return None
    
    # Save results
    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"{lang}_judged.parquet")
    passed_output_file = os.path.join(output_dir, f"{lang}_passed.parquet")
    os.makedirs(output_dir, exist_ok=True)

    passed_df = df[df['judge_result'] == 'PASSED']
    df.to_parquet(output_file)
    passed_df.to_parquet(passed_output_file)
    
    # Statistics
    passed_count = len(df[df['judge_result'] == 'PASSED'])
    failed_count = len(df[df['judge_result'] == 'FAILED'])
    
    logger.info(f"[{lang}] Saved {len(results)} judged records to {output_file}")
    logger.info(f"[{lang}] Saved {len(passed_df)} passed records to {passed_output_file}")
    logger.info(f"[{lang}] PASSED: {passed_count}, FAILED: {failed_count}")
    
    return {
        'total': len(results),
        'passed': passed_count,
        'failed': failed_count,
        'pass_rate': passed_count / len(results) if results else 0
    }


def discover_languages(batch_dir: str, codes_dir: str, judge_dir: str) -> list[str]:
    """
    Discover all languages that can be processed (must exist in all three directories).
    
    Args:
        batch_dir: Batch data directory
        codes_dir: Code generation result directory
        judge_dir: Verification result directory
    
    Returns:
        List of languages that can be processed
    """
    batch_langs = set()
    codes_langs = set()
    judge_langs = set()
    
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
    
    if os.path.exists(judge_dir):
        judge_langs = {
            d for d in os.listdir(judge_dir)
            if os.path.isdir(os.path.join(judge_dir, d))
        }
    
    # Intersection: must have batch data, code results and verification results
    common_langs = batch_langs & codes_langs & judge_langs
    
    logger.info(f"Batch directory languages: {sorted(batch_langs)}")
    logger.info(f"Code directory languages: {sorted(codes_langs)}")
    logger.info(f"Verification directory languages: {sorted(judge_langs)}")
    logger.info(f"Common languages: {sorted(common_langs)}")
    
    return sorted(list(common_langs))


# ============== Result Output ==============

def print_summary(all_stats: dict[str, Any], totals: dict[str, Any]) -> None:
    """Print processing summary information."""
    print("\n" + "=" * 80)
    print("Judgment result processing summary")
    print("=" * 80)
    
    print(f"\n{'Language':<20} {'Total':>10} {'Passed':>10} {'Failed':>10} {'Pass Rate':>12}")
    print("-" * 62)
    
    for lang, stats in sorted(all_stats.items()):
        print(
            f"{lang:<20} {stats['total']:>10} "
            f"{stats['passed']:>10} {stats['failed']:>10} "
            f"{stats['pass_rate']:>11.1%}"
        )
    
    print("-" * 62)
    print(
        f"{'Total':<20} {totals['total']:>10} "
        f"{totals['passed']:>10} {totals['failed']:>10} "
        f"{totals['pass_rate']:>11.1%}"
    )


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



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-language judgment result processing script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--batch_dir",
        default="dataset/synthetic_data/batch_data",
        help="Original code batch directory (output of prepare_batch_data.py)",
    )
    parser.add_argument(
        "-c", "--codes_dir",
        default="dataset/synthetic_data/codes",
        help="Code generation result directory (output of send_request_claude_stream.py)",
    )
    parser.add_argument(
        "-j", "--judge_dir",
        default="dataset/synthetic_data/judge_result",
        help="Verification result directory (output of send_request_openai.py verification mode)",
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="dataset/synthetic_data/judge_processed",
        help="Processing result output directory",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Specify the list of languages to process (default: all discovered languages)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge the results of all languages into a single file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Discover languages to process
    if args.languages:
        languages = args.languages
    else:
        languages = discover_languages(args.batch_dir, args.codes_dir, args.judge_dir)
    
    if not languages:
        logger.error("No languages found to process, please check input directories")
        return
    
    logger.info(f"Will process the following languages: {languages}")
    
    # Process each language
    all_stats: dict[str, Any] = {}
    all_results: list[dict] = []
    totals = {
        "total": 0,
        "passed": 0,
        "failed": 0,
    }
    
    for lang in tqdm(languages, desc="Processing languages", unit="lang"):
        result = process_language(
            lang=lang,
            batch_dir=args.batch_dir,
            codes_dir=args.codes_dir,
            judge_dir=args.judge_dir,
            output_dir=args.output_dir,
        )
        
        if result is None:
            continue
        
        all_stats[lang] = result
        totals["total"] += result["total"]
        totals["passed"] += result["passed"]
        totals["failed"] += result["failed"]
        
        # If merge is needed, read the recently saved file
        if args.merge:
            output_file = os.path.join(args.output_dir, f"{lang}_judged.parquet")
            if os.path.exists(output_file):
                df = pd.read_parquet(output_file)
                all_results.extend(df.to_dict('records'))
    
    # Calculate overall pass rate
    totals["pass_rate"] = totals["passed"] / totals["total"] if totals["total"] > 0 else 0
    
    # Print summary
    print_summary(all_stats, totals)
    
    # Save statistics
    save_stats(Path(args.output_dir) / "judge_stats.json", all_stats, totals)
    
    # If merge is needed, save the merged file
    if args.merge and all_results:
        merged_file = os.path.join(args.output_dir, "all_languages_judged.parquet")
        merged_df = pd.DataFrame(all_results)
        merged_df.to_parquet(merged_file)
        logger.info(f"Merged file saved to: {merged_file}")
        
        # Save PASSED records
        passed_df = merged_df[merged_df['judge_result'] == 'PASSED']
        passed_file = os.path.join(args.output_dir, "all_languages_passed.parquet")
        passed_df.to_parquet(passed_file)
        logger.info(f"PASSED records saved to: {passed_file} ({len(passed_df)} records)")


if __name__ == "__main__":
    main()

