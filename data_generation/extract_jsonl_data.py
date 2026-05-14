"""
aiXapply data generation pipeline.

Stage: prediction evaluation extraction.
Purpose: score prediction jsonl files and collect failing sample indices.
Inputs: prediction jsonl files from infer_openai.py.
Outputs: result_dict.json and optional diff artifacts.
"""
import json
import os
import sys
import difflib
import argparse
from pathlib import Path


def load_score_function():
    """Import lazily so CLI help does not require evaluation dependencies."""
    from training.rl.reward_function_rule_based_multi import compute_score_pygments

    return compute_score_pygments

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Defined as a function so that importing this module does not trigger
    ``argparse.parse_args()`` at module-load time. Callers should invoke
    ``build_arg_parser().parse_args()`` (or just rely on ``main()``).
    """
    p = argparse.ArgumentParser()
    # Output directory from infer_openai.py: dataset/synthetic_data/predictions/
    # Filename format: {model}_{lang}_{timestamp}.jsonl
    p.add_argument(
        "-i", "--input_files", nargs="*", default=None,
        help="Directly specify input file list (optional, choose either this or --predictions-dir)",
    )
    p.add_argument(
        "-d", "--predictions-dir",
        default="dataset/synthetic_data/predictions",
        help="Output directory of infer_openai.py, auto-discover .jsonl files within",
    )
    p.add_argument(
        "--languages", nargs="*", default=None,
        help="Specify languages to process (e.g., python java javascript), default processes all",
    )
    p.add_argument(
        "--model-prefix", default=None,
        help="Specify model prefix to filter files (e.g., glm-4.7), default processes all models",
    )
    p.add_argument(
        "--method", choices=["ast", "pygments", "both"], default="pygments",
        help="Method to use for extraction. ast and both evaluation is deprecated, only pygments is supported now",
    )
    p.add_argument(
        "--lang", default=None,
        help="Language type (deprecated, now auto-extracted from filename)",
    )
    p.add_argument(
        "--save-result", default="dataset/synthetic_data/predictions_result/result_dict.json",
        help="File path to save results",
    )
    p.add_argument(
        "--diff-output-dir", default="dataset/synthetic_data/predictions_compared",
        help="Directory to save optional diff artifacts",
    )
    p.add_argument("--save-diff", action="store_true", default=False)
    return p


PREDICTION_WRAPPER_MARKERS = (
    "<update_code>",
    "</update_code>",
    "<updated-code>",
    "</updated-code>",
    "<update_file>\n",
    "<update_file>",
    "</update_file>",
    "</think>\n",
    "</think>",
)


def clean_prediction_content(content) -> str:
    """Remove model wrapper tags while preserving the generated file body."""
    if not isinstance(content, str):
        return ""

    for marker in PREDICTION_WRAPPER_MARKERS:
        content = content.replace(marker, "")
    return content.replace("\u00a0", " ")


def extract_lang_from_filename(filename: str) -> str:
    """
    extract language from filename
    support two filename formats:
    1. {model}_{lang}_{timestamp}.jsonl (timestamp: YYYYMMDD_HHMMSS)
        example: glm-4.7_python_20251230_200411.jsonl -> python
    2. {model}_{lang}_{frequency}.jsonl
        example: deepseek-v3-2-251201_python_1.jsonl -> python
    """
    # remove .jsonl suffix
    base_name = filename.replace('.jsonl', '')
    parts = base_name.split('_')
    
    if len(parts) < 2:
        return "python"
    
    # check if is old format (timestamp)
    # old format: last two parts are YYYYMMDD and HHMMSS
    if len(parts) >= 4:
        last_part = parts[-1]
        second_last = parts[-2]
        # if last part is 6 digits, second last part is 8 digits, then it is old format
        if (len(last_part) == 6 and last_part.isdigit() and 
            len(second_last) == 8 and second_last.isdigit()):
            # old format: language is third last
            return parts[-3]
    
    # new format: {model}_{lang}_{frequency}
    # language is second last
    return parts[-2]


def discover_prediction_files(predictions_dir: str, languages: list = None, model_prefix: str = None) -> list:
    """
    discover .jsonl files in predictions directory
    
    Args:
        predictions_dir: predictions directory
        languages: language list to process, None means all languages
        model_prefix: model prefix filter, None means all models
    
    Returns:
        list of (file path, language)
    """
    predictions_path = Path(predictions_dir)
    if not predictions_path.exists():
        print(f"Warning: Predictions directory not found: {predictions_dir}")
        return []
    
    # find all .jsonl files
    all_files = list(predictions_path.glob("*.jsonl"))
    
    result = []
    for file_path in sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True):
        filename = file_path.name
        lang = extract_lang_from_filename(filename)
        
        # if language list is specified, only process specified languages
        if languages is not None and lang not in languages:
            continue
        
        # if model prefix is specified, only process matching models
        if model_prefix is not None and not filename.startswith(model_prefix):
            continue
        
        result.append((str(file_path), lang))
    
    return result


def extract_data(args, input_file, lang=None):
    """
    extract and evaluate predictions
    
    Args:
        args: command line arguments
        input_file: input file path
        lang: language type (for pygments method)
    """
    # if language is not specified, extract from filename
    if lang is None:
        lang = extract_lang_from_filename(os.path.basename(input_file))
    compute_score_pygments = load_score_function()
    
    output_base_dir = str(Path(args.diff_output_dir) / Path(input_file).stem)
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return None, None

    # Create base output directory
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"Created directory: {output_base_dir}")

    total, correct = 0, 0
    errors_list = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON on line {line_number + 1}")
                    continue

                # get language from record (if specified)
                record_lang = data.get('language', lang)

                # get index as folder name
                idx = data.get('index')
                if idx is None:
                    # if index field is not specified, use line number
                    idx = line_number
                
                # create folder for corresponding index
                target_dir = os.path.join(output_base_dir, str(idx))
                os.makedirs(target_dir, exist_ok=True)

                # define fields to save and corresponding file names
                file_mapping = {
                    'prediction': 'prediction.py',
                    'ground_truth': 'ground_truth.py',
                    'original_code': 'original_code.py',
                    'update_snippet': 'update_snippet.py'
                }

                # flag_save_diff = args.save_diff and random.random() < 0.15
                flag_save_diff = args.save_diff
                # write to file
                for key, filename in file_mapping.items():
                    content = data.get(key)
                    if key == 'prediction':
                        content = clean_prediction_content(data.get("prediction"))
                        prediction = content
                    # if key == "update_snippet":
                    #     pass
                    if content is not None:
                        if flag_save_diff:
                            file_path = os.path.join(target_dir, filename)
                            with open(file_path, 'w', encoding='utf-8') as out_f:
                                out_f.write(content)

                score = compute_score_pygments(prediction, data['ground_truth'], record_lang)

                if score == 0:
                    errors_list.append(idx)
                    if flag_save_diff:
                        diff = difflib.unified_diff(
                            data['ground_truth'].splitlines(keepends=True),
                            prediction.splitlines(keepends=True),
                            fromfile="ground_truth",
                            tofile="prediction",
                            n=2,
                        )
                        file_path = os.path.join(target_dir, 'diff.txt')
                        print(f"Saved diff to {file_path}")
                        with open(file_path, 'w', encoding='utf-8') as out_f:
                            for i, line in enumerate(diff):
                                out_f.write(line)
                correct += score
                total += 1
                # if (line_number + 1) % 10 == 0:
                #     print(f"Processed {line_number + 1} records...")
        
        if total == 0:
            print(f"Warning: No records found in {input_file}")
            return [], 0.0
            
        Accuracy = correct/total
        print(f"\nExtraction complete! Data saved to '{output_base_dir}/'")
        print(f"Language: {lang}, Method: {args.method}")
        print(f"Total: {total}, Correct: {correct}, Accuracy: {Accuracy:.4f}")
        print(f"Errors: {errors_list}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], 0.0
    return errors_list, Accuracy

def main() -> None:
    args = build_arg_parser().parse_args()

    # determine input file list
    if args.input_files:
        input_files_with_lang = [
            (f, extract_lang_from_filename(os.path.basename(f))) for f in args.input_files
        ]
    else:
        input_files_with_lang = discover_prediction_files(
            args.predictions_dir,
            args.languages,
            args.model_prefix,
        )

    if not input_files_with_lang:
        print("Error: No prediction files found!")
        print(f"  - predictions_dir: {args.predictions_dir}")
        print(f"  - languages:       {args.languages}")
        print(f"  - model_prefix:    {args.model_prefix}")
        sys.exit(1)

    print("=" * 80)
    print(f"Discovered {len(input_files_with_lang)} prediction file(s):")
    print("=" * 80)
    for file_path, lang in input_files_with_lang:
        print(f"  [{lang}] {file_path}")
    print("=" * 80 + "\n")

    method_list = ["ast", "pygments"] if args.method == "both" else [args.method]

    result_dict = {}
    for method in method_list:
        args.method = method
        result_dict[method] = {}
        for input_file, lang in input_files_with_lang:
            print(f"\n{'=' * 60}")
            print(f"Processing: {input_file}")
            print(f"Language:   {lang}, Method: {method}")
            print(f"{'=' * 60}")

            errors_list, accuracy = extract_data(args, input_file, lang)
            if errors_list is not None:
                key = input_file.split("/")[-1].replace(".jsonl", "")
                result_dict[method][key] = {
                    "language": lang,
                    "errors_list": errors_list,
                    "Accuracy": accuracy,
                }

    print("\n" + "=" * 80)
    print("Summary results")
    print("=" * 80)

    for method, results in result_dict.items():
        print(f"\nMethod: {method}")
        print("-" * 60)
        print(f"{'File name':<50} {'Language':<10} {'Accuracy':<10}")
        print("-" * 60)

        total_files = 0
        total_acc = 0.0
        for file_name, data in results.items():
            lang = data.get("language", "unknown")
            acc = data.get("Accuracy", 0.0)
            print(f"{file_name:<50} {lang:<10} {acc:.4f}")
            total_files += 1
            total_acc += acc

        if total_files > 0:
            print("-" * 60)
            print(f"{'Average':<50} {'':<10} {total_acc / total_files:.4f}")

    print("\n" + "=" * 80)
    print("Detailed results:")
    print(json.dumps(result_dict, indent=2, ensure_ascii=False))

    if args.save_result:
        os.makedirs(os.path.dirname(args.save_result), exist_ok=True)
        with open(args.save_result, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.save_result}")


if __name__ == "__main__":
    main()
