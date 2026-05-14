import argparse
import difflib
import json
import os
import time
from collections import Counter

from experiments.evaluation.error_classifier import ClassifierOptions, classify_record
from experiments.evaluation.error_types import ErrorType as ErrorTypeEnum
from experiments.evaluation.llm_judge import LLMConfig
from training.rl.reward_function_rule_based_multi import (
    compute_score_pygments,
    strip_update_tags,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Wrapped in a function so importing this module does not trigger argparse
    side effects at module-load time.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i",
        "--input_files",
        nargs="+",
        required=True,
        help="Prediction jsonl files to evaluate.",
    )

    p.add_argument("--save_all", action="store_true", default=False)
    p.add_argument(
        "--classify_errors",
        action="store_true",
        default=False,
        help="Further classify wrong samples into error types",
    )
    p.add_argument(
        "--classify_debug",
        action="store_true",
        default=False,
        help="Include diff/features in classification output (larger output)",
    )

    # LLM (optional): semantic judging of error types
    p.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help="Enable LLM-based semantic judging (except OUTPUT_INVALID)",
    )
    p.add_argument("--llm_model", default=os.getenv("OPENAI_MODEL", "deepseek-v3-2-251201"))
    p.add_argument("--llm_api_key", default=os.getenv("OPENAI_API_KEY", ""))
    p.add_argument(
        "--llm_base_url",
        default=os.getenv("OPENAI_BASE_URL")
        or os.getenv("BASE_URL")
        or "https://api.openai.com/v1",
    )
    p.add_argument("--llm_max_tokens", type=int, default=8192)
    p.add_argument("--llm_timeout_s", type=float, default=360.0)
    p.add_argument(
        "--rule_skip_llm_threshold",
        type=float,
        default=0.9,
        help="Allow skipping LLM only when rule confidence >= this threshold",
    )
    return p
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
    # Config formats
    "json": "json",
    "yaml": "yaml",
    "xml": "xml",
    "sql": "sql",
    "ini": "ini",
    # Documents / markup
    "markdown": "md",
    "text": "txt",
    "html": "html",
    "restructuredtext": "rst",
    # DevOps / build scripts
    "shell": "sh",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
}


REQUIRED_RECORD_KEYS: tuple[str, ...] = (
    "index",
    "prediction",
    "elapsed_time",
    "ground_truth",
    "original_code",
    "update_snippet",
    "language",
)

REQUIRED_NON_EMPTY_STRING_KEYS: tuple[str, ...] = (
    "prediction",
    "ground_truth",
    "original_code",
    "update_snippet",
    "language",
)


def validate_record_data(data) -> list[str]:
    """Validate a single jsonl record has required keys and non-empty key string fields."""
    if not isinstance(data, dict):
        return [f"record is not a JSON object (got {type(data).__name__})"]

    errors: list[str] = []
    missing = [k for k in REQUIRED_RECORD_KEYS if k not in data]
    if missing:
        errors.append(f"missing keys: {missing}")

    for k in REQUIRED_NON_EMPTY_STRING_KEYS:
        if k not in data:
            continue
        v = data.get(k)
        if not isinstance(v, str):
            errors.append(f"{k} is not a string")
        elif len(v.strip()) == 0:
            errors.append(f"{k} is empty")

    return errors


def extract_data(args, input_file):
    output_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'results', input_file.split('/')[-1].replace('.jsonl', ''))
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create base output directory
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"Created directory: {output_base_dir}")

    total_lang = {}
    errors_list = []
    correct_lang = {}
    total, correct = 0, 0
    final_error_type_counter: Counter[str] = Counter()
    rule_error_type_counter: Counter[str] = Counter()
    llm_error_type_counter: Counter[str] = Counter()
    used_llm_count = 0
    llm_success_count = 0

    # Classification output (one jsonl per input_file)
    classify_writer = None
    classify_path = None
    classify_options = None
    if args.classify_errors:
        os.makedirs(output_base_dir, exist_ok=True)
        classify_path = os.path.join(output_base_dir, "error_classification.jsonl")
        classify_writer = open(classify_path, "w", encoding="utf-8")
        llm_cfg = LLMConfig(
            enabled=bool(args.llm),
            model=args.llm_model,
            api_key=args.llm_api_key or None,
            base_url=args.llm_base_url,
            max_tokens=args.llm_max_tokens,
            timeout_s=args.llm_timeout_s,
        )
        classify_options = ClassifierOptions(
            llm=llm_cfg,
            debug=bool(args.classify_debug),
            rule_skip_llm_conf_threshold=float(args.rule_skip_llm_threshold),
        )

    with open(input_file, 'r', encoding='utf-8-sig') as f:
        for line_number, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_number + 1}")
                continue

            validation_errors = validate_record_data(data)
            if validation_errors:
                print(
                    f"Skipping invalid record on line {line_number + 1}: "
                    + "; ".join(validation_errors)
                )
                continue

            # Note: read prediction first, then score (avoid undefined 'prediction')
            prediction = data.get("prediction") or ""
            score = compute_score_pygments(prediction, data['ground_truth'], data.get('language', None))
            prediction = strip_update_tags(prediction)

            target_dir = None
            idx = None
            if score == 0:
                # Use index as the folder name (only needed for wrong samples)
                idx = data.get('index')
                if idx is None:
                    idx = line_number

                # Persist per-sample artifacts only when save_all is enabled
                if args.save_all:
                    target_dir = os.path.join(output_base_dir, str(idx))
                    os.makedirs(target_dir, exist_ok=True)

                    # Clean up legacy files to avoid mixing .py/.js in the same folder
                    for legacy_name in [
                        "prediction.py",
                        "ground_truth.py",
                        "original_code.py",
                        "update_snippet.py",
                        "diff.txt",
                    ]:
                        legacy_path = os.path.join(target_dir, legacy_name)
                        if os.path.exists(legacy_path):
                            try:
                                os.remove(legacy_path)
                            except OSError as exc:
                                # Stale legacy file we could not clean up; not
                                # fatal but worth surfacing in --log-level=DEBUG.
                                print(
                                    f"warning: failed to remove legacy {legacy_path}: {exc}"
                                )

                    lang_key = (data.get("language") or "").lower().strip()
                    ext = EXTENSION_MAP.get(lang_key, "txt")

                    # Map fields to output filenames
                    file_mapping = {
                        'prediction': f'prediction.{ext}',
                        'ground_truth': f'ground_truth.{ext}',
                        'original_code': f'original_code.{ext}',
                        'update_snippet': f'update_snippet.{ext}'
                    }

                    # Write files
                    for key, filename in file_mapping.items():
                        content = data.get(key)
                        if key == 'prediction':
                            content = strip_update_tags(content)
                        if content is not None:
                            file_path = os.path.join(target_dir, filename)
                            with open(file_path, 'w', encoding='utf-8') as out_f:
                                out_f.write(content)

            if score == 0:
                errors_list.append(idx)

                # Error type classification (optional)
                if classify_writer is not None and classify_options is not None:
                    try:
                        cls = classify_record(data, classify_options)
                    except Exception as e:
                        # Surface the traceback so the user notices that the
                        # rule-based classifier is silently failing on real data.
                        import traceback as _tb

                        print(
                            f"warning: classifier crashed on index {data.get('index')}: {e}\n"
                            f"{_tb.format_exc()}"
                        )
                        cls = {
                            "error_type": "CLASSIFIER_CRASH",
                            "confidence": 0.0,
                            "needs_review": True,
                            "reason_brief": [f"classifier_exception: {e}"],
                        }
                    final_error_type_counter[str(cls.get("error_type"))] += 1
                    rule_error_type_counter[str(cls.get("rule_error_type"))] += 1
                    if cls.get("used_llm"):
                        used_llm_count += 1
                    if cls.get("llm_error_type"):
                        llm_success_count += 1
                        llm_error_type_counter[str(cls.get("llm_error_type"))] += 1
                    out_row = {
                        "index": idx,
                        "language": data.get("language"),
                        # Final result (usually from LLM; can skip if rule is high-confidence)
                        "error_type": cls.get("error_type"),
                        "confidence": cls.get("confidence"),
                        "needs_review": cls.get("needs_review"),
                        "reason_brief": cls.get("reason_brief"),
                        # Rule-side result (always output for comparison)
                        "rule_error_type": cls.get("rule_error_type"),
                        "rule_confidence": cls.get("rule_confidence"),
                        "rule_reason_brief": cls.get("rule_reason_brief"),
                        "rule_deterministic": cls.get("rule_deterministic"),
                        "rule_can_skip_llm": cls.get("rule_can_skip_llm"),
                        # LLM-side result (when enabled and rule is not eligible to skip)
                        "used_llm": cls.get("used_llm"),
                        "llm_error_type": cls.get("llm_error_type"),
                        "llm_confidence": cls.get("llm_confidence"),
                        "llm_reason_brief": cls.get("llm_reason_brief"),
                    }
                    if args.classify_debug:
                        out_row["debug"] = cls.get("debug")
                    classify_writer.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    classify_writer.flush()

                    # If saving per-sample dir, also write classification.json for quick inspection
                    if args.save_all and target_dir is not None:
                        cls_path = os.path.join(target_dir, "classification.json")
                        with open(cls_path, "w", encoding="utf-8") as cf:
                            cf.write(json.dumps(out_row, ensure_ascii=False, indent=2))

                if args.save_all and target_dir is not None:
                    original_code = data.get("original_code") or ""
                    ground_truth = data.get("ground_truth") or ""

                    diffs = {
                        # original -> prediction (actual applied diff)
                        "diff_original_to_prediction.diff": difflib.unified_diff(
                            original_code.splitlines(keepends=True),
                            prediction.splitlines(keepends=True),
                            fromfile="original",
                            tofile="prediction",
                            n=2,
                        ),
                        # original -> ground_truth (expected diff)
                        "diff_original_to_ground_truth.diff": difflib.unified_diff(
                            original_code.splitlines(keepends=True),
                            ground_truth.splitlines(keepends=True),
                            fromfile="original",
                            tofile="ground_truth",
                            n=2,
                        ),
                        # ground_truth -> prediction (prediction vs GT)
                        "diff_ground_truth_to_prediction.diff": difflib.unified_diff(
                            ground_truth.splitlines(keepends=True),
                            prediction.splitlines(keepends=True),
                            fromfile="ground_truth",
                            tofile="prediction",
                            n=2,
                        ),
                    }

                    for name, diff_iter in diffs.items():
                        file_path = os.path.join(target_dir, name)
                        with open(file_path, "w", encoding="utf-8") as out_f:
                            for line in diff_iter:
                                out_f.write(line)
                        # print(f"Saved diff to {file_path}")
            correct += score
            total += 1
            correct_lang[data.get('language', None)] = correct_lang.get(data.get('language', None), 0) + score
            total_lang[data.get('language', None)] = total_lang.get(data.get('language', None), 0) + 1
            if (line_number + 1) % 40 == 0:
                print(f"Processed {line_number + 1} records...")
    if total == 0:
        raise ValueError(f"No valid records found in {input_file}")

    Accuracy = correct/total
    Acc_lang = {lang: correct_lang[lang]/total_lang[lang] for lang in correct_lang}
    print(f"\nExtraction complete! Data saved to '{output_base_dir}/'")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {Accuracy:.4f}")
    print(f"Accuracy by language: {Acc_lang}")
    print(f"Errors: {errors_list}")
    if classify_writer is not None:
        classify_writer.close()
        print(f"Saved error classification to {classify_path}")

    # Classification stats: per-type ratios (denominator = wrong samples)
    error_type_counts = None
    error_type_ratios = None
    rule_error_type_counts = None
    rule_error_type_ratios = None
    llm_error_type_counts = None
    llm_error_type_ratios = None
    llm_usage = None
    if args.classify_errors:
        canonical = [e.value for e in ErrorTypeEnum]
        # final
        error_type_counts = {t: int(final_error_type_counter.get(t, 0)) for t in canonical}
        for t, c in final_error_type_counter.items():
            if t not in error_type_counts:
                error_type_counts[t] = int(c)
        denom = sum(error_type_counts.values())
        error_type_ratios = {t: (c / denom if denom else 0.0) for t, c in error_type_counts.items()}

        # rule
        rule_error_type_counts = {t: int(rule_error_type_counter.get(t, 0)) for t in canonical}
        for t, c in rule_error_type_counter.items():
            if t not in rule_error_type_counts:
                rule_error_type_counts[t] = int(c)
        denom_rule = sum(rule_error_type_counts.values())
        rule_error_type_ratios = {
            t: (c / denom_rule if denom_rule else 0.0) for t, c in rule_error_type_counts.items()
        }

        # llm (only count cases where llm_error_type exists)
        llm_error_type_counts = {t: int(llm_error_type_counter.get(t, 0)) for t in canonical}
        for t, c in llm_error_type_counter.items():
            if t not in llm_error_type_counts:
                llm_error_type_counts[t] = int(c)
        denom_llm = sum(llm_error_type_counts.values())
        llm_error_type_ratios = {t: (c / denom_llm if denom_llm else 0.0) for t, c in llm_error_type_counts.items()}

        llm_usage = {
            "used_llm_count": int(used_llm_count),
            "llm_success_count": int(llm_success_count),
            "llm_enabled": bool(args.llm),
            "rule_skip_llm_threshold": float(args.rule_skip_llm_threshold),
        }

    return (
        errors_list,
        Accuracy,
        Acc_lang,
        error_type_counts,
        error_type_ratios,
        rule_error_type_counts,
        rule_error_type_ratios,
        llm_error_type_counts,
        llm_error_type_ratios,
        llm_usage,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    result_dict = {}
    for input_file in args.input_files:
        (
            errors_list,
            Accuracy,
            Acc_lang,
            error_type_counts,
            error_type_ratios,
            rule_error_type_counts,
            rule_error_type_ratios,
            llm_error_type_counts,
            llm_error_type_ratios,
            llm_usage,
        ) = extract_data(args, input_file)
        input_key = input_file.split("/")[-1].replace(".jsonl", "")
        per_file_result = {
            "errors_list": errors_list,
            "Accuracy": Accuracy,
            "Acc_lang": Acc_lang,
            # final
            "error_type_counts": error_type_counts,
            "error_type_ratios": error_type_ratios,
            # rule vs llm comparison
            "rule_error_type_counts": rule_error_type_counts,
            "rule_error_type_ratios": rule_error_type_ratios,
            "llm_error_type_counts": llm_error_type_counts,
            "llm_error_type_ratios": llm_error_type_ratios,
            "llm_usage": llm_usage,
        }
        result_dict[input_key] = per_file_result

        # Write final per-input statistics json under this input's output_base_dir
        output_base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results",
            input_key,
        )
        os.makedirs(output_base_dir, exist_ok=True)
        final_stats_path = os.path.join(output_base_dir, "final_stats.json")
        with open(final_stats_path, "w", encoding="utf-8") as f:
            json.dump(per_file_result, f, ensure_ascii=False, indent=2)
    from pprint import pprint
    for input_key in result_dict:
        result_dict[input_key].pop("errors_list")
    pprint(result_dict)


if __name__ == "__main__":
    main()