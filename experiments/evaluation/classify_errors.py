from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from experiments.evaluation.error_classifier import ClassifierOptions, classify_record
from experiments.evaluation.llm_judge import LLMConfig
from training.rl.reward_function_rule_based_multi import compute_score_pygments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify apply-model error types for WRONG examples.")
    p.add_argument(
        "-i",
        "--input_files",
        nargs="+",
        required=True,
        help="Input jsonl files containing {prediction, ground_truth, original_code, language?}",
    )
    p.add_argument(
        "-o",
        "--output_dir",
        default="predictions/classified",
        help="Output directory for classified jsonl.",
    )
    p.add_argument(
        "--only_errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only classify wrong samples (default: true). Use --no-only_errors to classify all samples.",
    )
    p.add_argument("--max_samples", type=int, default=0, help="Optional max number of classified samples.")
    p.add_argument("--debug", action="store_true", default=False, help="Include diffs/features in output.")

    # LLM config
    p.add_argument("--llm", action="store_true", default=False, help="Enable LLM semantic judging.")
    p.add_argument("--llm_model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--llm_api_key", default=os.getenv("OPENAI_API_KEY", ""))
    p.add_argument("--llm_base_url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--llm_max_tokens", type=int, default=512)
    p.add_argument("--llm_timeout_s", type=float, default=120.0)
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = LLMConfig(
        enabled=bool(args.llm),
        model=args.llm_model,
        api_key=args.llm_api_key or None,
        base_url=args.llm_base_url,
        max_tokens=args.llm_max_tokens,
        timeout_s=args.llm_timeout_s,
    )
    options = ClassifierOptions(llm=llm_cfg, debug=bool(args.debug))

    total_seen = 0
    total_classified = 0
    type_counter: Counter[str] = Counter()
    needs_review_counter = 0

    for in_file in args.input_files:
        in_path = Path(in_file)
        if not in_path.exists():
            print(f"[WARN] missing input: {in_file}", flush=True)
            continue

        out_path = out_dir / (in_path.stem + ".classified.jsonl")
        with out_path.open("w", encoding="utf-8") as out_f:
            for rec in iter_jsonl(in_path):
                total_seen += 1
                if args.max_samples and total_classified >= args.max_samples:
                    break

                prediction = rec.get("prediction") or ""
                gt = rec.get("ground_truth") or ""
                lang = rec.get("language")

                if args.only_errors:
                    try:
                        score = compute_score_pygments(prediction, gt, lang)
                    except Exception:
                        score = 0.0
                    if score == 1.0:
                        continue

                result = classify_record(rec, options)
                out = {
                    "index": rec.get("index"),
                    "language": lang,
                    "error_type": result["error_type"],
                    "confidence": result.get("confidence"),
                    "needs_review": result.get("needs_review"),
                    "reason_brief": result.get("reason_brief"),
                    "rule_error_type": result.get("rule_error_type"),
                    "rule_confidence": result.get("rule_confidence"),
                    "rule_reason_brief": result.get("rule_reason_brief"),
                    "rule_deterministic": result.get("rule_deterministic"),
                    "rule_can_skip_llm": result.get("rule_can_skip_llm"),
                    "used_llm": result.get("used_llm"),
                    "llm_error_type": result.get("llm_error_type"),
                    "llm_confidence": result.get("llm_confidence"),
                    "llm_reason_brief": result.get("llm_reason_brief"),
                }
                if args.debug:
                    out["debug"] = result.get("debug")

                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                total_classified += 1
                type_counter[out["error_type"]] += 1
                if out.get("needs_review"):
                    needs_review_counter += 1

        print(f"[OK] wrote: {out_path}", flush=True)

    print("==== Summary ====")
    print(f"total_seen={total_seen}")
    print(f"total_classified={total_classified}")
    print(f"needs_review={needs_review_counter}")
    print(dict(type_counter))


if __name__ == "__main__":
    main()

