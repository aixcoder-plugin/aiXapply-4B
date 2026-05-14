from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent

from experiments.evaluation.diff_features import compute_diff_feature_summary, unified_diff_text
from experiments.evaluation.llm_judge import LLMConfig, call_llm_chat
from experiments.evaluation.update_file_parser import check_output_invalid
from training.rl.reward_function_rule_based_multi import compute_score_pygments

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    # Optional dependency; environment variables can still be provided by the shell.
    pass


SYSTEM_PROMPT = """You are a strict judge for apply-task correctness.
You will be given evidence about an apply task:
- original full file
- update_snippet (the requested partial update)
- ground_truth (the expected full updated file)
- prediction (the model-produced full updated file)
- diffs and automatic signals

Your job is to determine whether the prediction is an acceptable final full-file result.

Judging rules:
- Use ground_truth as a strong reference, but do not require byte-for-byte identity when the prediction still correctly applies the requested change.
- Ignore pure whitespace-only or formatting-only differences.
- Do NOT ignore missing required changes, missing imports, wrong insertion location, extra side effects, broken wrappers, placeholder leakage, or behavior-changing differences.
- Comment or docstring wording differences may be acceptable only when they are clearly not part of the requested change.
- If the prediction cannot be used as a valid full-file output, mark it incorrect.

Return ONLY valid JSON. No markdown, no explanation outside JSON.
"""

VALID_VERDICTS = {"correct", "incorrect", "needs_review"}
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "fastapply-test"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "apply_correctness_judge"


FIELD_CANDIDATES: Dict[str, List[str]] = {
    "index": ["index", "id", "sample_id"],
    "language": ["language", "lang", "file_language"],
    "original_code": ["original_code", "old_code", "source_file", "source", "original"],
    "update_snippet": ["update_snippet", "udiff", "patch", "instruction", "edit", "message"],
    "ground_truth": ["ground_truth", "new_code", "final_code", "target_code", "expected_code"],
    "prediction": ["prediction", "output", "pred", "completion", "response", "model_output"],
    "path": ["path", "file_path"],
    "repo": ["repo", "repository"],
    "commit": ["commit", "sha"],
    "message": ["message", "commit_message"],
}


@dataclass
class ApplySample:
    source_file: str
    line_no: int
    index: str
    language: str
    original_code: str
    update_snippet: str
    ground_truth: str
    prediction_raw: str
    path: str
    repo: str
    commit: str
    message: str
    extracted_fields: Dict[str, Optional[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call an OpenAI-compatible LLM to judge whether apply-task predictions in jsonl files "
            "are correct, then write result/reason jsonl and summary files."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing input jsonl files.",
    )
    parser.add_argument(
        "--input-pattern",
        default="*.jsonl",
        help="Glob pattern used under --input-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for judged jsonl and summary files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Only process the first N matched jsonl files (0 means all).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Only process the first N samples across all files (0 means all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Do not call the LLM; write fallback judgments to validate the pipeline.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Write extra debug fields such as raw LLM output and diff strings.",
    )
    parser.add_argument(
        "--diff-context-lines",
        type=int,
        default=2,
        help="Context lines used when building diff evidence.",
    )
    parser.add_argument(
        "--diff-max-lines",
        type=int,
        default=400,
        help="Maximum diff lines kept for each diff section sent to the LLM.",
    )
    parser.add_argument(
        "--max-update-snippet-chars",
        type=int,
        default=12000,
        help="Maximum characters kept for update_snippet in the prompt.",
    )

    parser.add_argument("--llm_model", default=os.getenv("OPENAI_MODEL") or os.getenv("MODEL") or "deepseek-v3-2-251201")
    parser.add_argument(
        "--llm_api_key",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or "EMPTY",
        help="API key for the OpenAI-compatible endpoint. Dummy values are fine for many local endpoints.",
    )
    parser.add_argument(
        "--llm_base_url",
        default=os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or "https://api.openai.com/v1",
    )
    parser.add_argument("--llm_max_tokens", type=int, default=32768)
    parser.add_argument("--llm_timeout_s", type=float, default=120.0)
    parser.add_argument("--llm_temperature", type=float, default=0.3)
    parser.add_argument("--llm_top_p", type=float, default=0.8)
    parser.add_argument(
        "--llm_extra_body",
        default=None,
        help='Optional JSON string merged into the OpenAI-compatible request body.',
    )
    return parser.parse_args()


def _parse_extra_body(extra_body: Optional[str]) -> Dict[str, Any]:
    if not extra_body:
        return {}
    try:
        obj = json.loads(extra_body)
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        raise ValueError(f"--llm_extra_body must be valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("--llm_extra_body must be a JSON object.")
    return obj


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _pick_first(record: Dict[str, Any], logical_name: str) -> Tuple[str, Optional[str]]:
    for key in FIELD_CANDIDATES[logical_name]:
        if key in record and record[key] is not None:
            return _stringify(record[key]), key
    return "", None


def _iter_jsonl(path: Path) -> Iterator[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                yield line_no, None, f"input_json_parse_error: {exc}"
                continue
            if not isinstance(obj, dict):
                yield line_no, None, "input_json_not_object"
                continue
            yield line_no, obj, None


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    return (
        text[:keep_head]
        + f"\n... (content truncated, total_chars={len(text)}) ...\n"
        + text[-keep_tail:]
    )


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()

    # Fast path: the entire response is already valid JSON. If parsing fails
    # we deliberately fall through to the brace-scanning fallback below;
    # JSONDecodeError here is expected and does not indicate a bug.
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = stripped[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def _normalize_judge_output(raw_response: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    obj = _extract_first_json_object(raw_response)
    if obj is None:
        return None, "judge_json_parse_failed"

    verdict = str(obj.get("verdict") or "").strip().lower()
    if verdict not in VALID_VERDICTS:
        return None, f"invalid_verdict: {verdict}"

    try:
        confidence = float(obj.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reason = str(obj.get("reason") or "").strip()
    key_points_raw = obj.get("key_points")
    if isinstance(key_points_raw, list):
        key_points = [str(item)[:300] for item in key_points_raw if str(item).strip()][:3]
    elif isinstance(key_points_raw, str) and key_points_raw.strip():
        key_points = [key_points_raw[:300]]
    else:
        key_points = []

    if not reason and key_points:
        reason = " | ".join(key_points)
    if not reason:
        return None, "missing_reason"

    normalized = {
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason[:1000],
        "key_points": key_points,
        "matches_ground_truth": _coerce_optional_bool(obj.get("matches_ground_truth")),
        "requested_change_applied": _coerce_optional_bool(obj.get("requested_change_applied")),
    }
    return normalized, None


def _build_sample(source_file: Path, line_no: int, record: Dict[str, Any]) -> ApplySample:
    original_code, original_code_key = _pick_first(record, "original_code")
    update_snippet, update_snippet_key = _pick_first(record, "update_snippet")
    ground_truth, ground_truth_key = _pick_first(record, "ground_truth")
    prediction_raw, prediction_key = _pick_first(record, "prediction")
    language, language_key = _pick_first(record, "language")
    path_text, path_key = _pick_first(record, "path")
    repo_text, repo_key = _pick_first(record, "repo")
    commit_text, commit_key = _pick_first(record, "commit")
    message_text, message_key = _pick_first(record, "message")
    index_text, index_key = _pick_first(record, "index")

    if not index_text:
        index_text = str(line_no)

    return ApplySample(
        source_file=str(source_file),
        line_no=line_no,
        index=index_text,
        language=language,
        original_code=original_code,
        update_snippet=update_snippet,
        ground_truth=ground_truth,
        prediction_raw=prediction_raw,
        path=path_text,
        repo=repo_text,
        commit=commit_text,
        message=message_text,
        extracted_fields={
            "index": index_key,
            "language": language_key,
            "original_code": original_code_key,
            "update_snippet": update_snippet_key,
            "ground_truth": ground_truth_key,
            "prediction": prediction_key,
            "path": path_key,
            "repo": repo_key,
            "commit": commit_key,
            "message": message_key,
        },
    )


def _build_compact_feature_summary(feature_summary: Dict[str, Any]) -> Dict[str, Any]:
    if not feature_summary:
        return {}
    return {
        "no_op_score": feature_summary.get("no_op_score"),
        "patch_match_score": feature_summary.get("patch_match_score"),
        "expected_edit_count": feature_summary.get("expected_edit_count"),
        "actual_edit_count": feature_summary.get("actual_edit_count"),
        "side_effect_opcode_count": feature_summary.get("side_effect_opcode_count"),
        "patch_area_opcode_count": feature_summary.get("patch_area_opcode_count"),
        "missing_expected_edit_ids": feature_summary.get("missing_expected_edit_ids"),
        "near_miss_expected_edit_ids": feature_summary.get("near_miss_expected_edit_ids"),
        "misplaced_expected_edit_ids": feature_summary.get("misplaced_expected_edit_ids"),
        "expected_edits_preview": feature_summary.get("expected_edits", [])[:5],
        "side_effect_opcodes_preview": feature_summary.get("side_effect_opcodes_preview", [])[:10],
    }


def _build_messages(
    *,
    sample: ApplySample,
    expected_diff: str,
    actual_diff: str,
    prediction_vs_ground_truth_diff: str,
    auto_signals: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    update_snippet = sample.update_snippet if sample.update_snippet else "<missing update_snippet>"
    update_snippet = _truncate_text(update_snippet, args.max_update_snippet_chars)

    user_prompt = f"""Judge whether the apply-task prediction is correct.

Metadata:
- sample_index: {sample.index}
- input_file: {sample.source_file}
- input_line: {sample.line_no}
- language: {sample.language or "unknown"}
- path: {sample.path or "<unknown>"}
- repo: {sample.repo or "<unknown>"}
- commit: {sample.commit or "<unknown>"}

update_snippet:
<<<UPDATE_SNIPPET
{update_snippet}
UPDATE_SNIPPET

expected_diff (original -> ground_truth):
<<<EXPECTED_DIFF
{expected_diff or "<empty diff>"}
EXPECTED_DIFF

actual_diff (original -> prediction):
<<<ACTUAL_DIFF
{actual_diff or "<empty diff>"}
ACTUAL_DIFF

prediction_vs_ground_truth_diff:
<<<PREDICTION_VS_GROUND_TRUTH_DIFF
{prediction_vs_ground_truth_diff or "<empty diff>"}
PREDICTION_VS_GROUND_TRUTH_DIFF

automatic_signals (JSON):
<<<AUTO_SIGNALS
{json.dumps(auto_signals, ensure_ascii=False)}
AUTO_SIGNALS

Output requirements:
- Output ONLY a JSON object.
- JSON schema:
  {{
    "verdict": "correct" | "incorrect" | "needs_review",
    "confidence": <number between 0 and 1>,
    "reason": "<one concise reason>",
    "key_points": ["<1-3 short evidence points>"],
    "matches_ground_truth": true | false | null,
    "requested_change_applied": true | false | null
  }}

Decision standard:
- "correct": the prediction is an acceptable full-file result for the apply task.
- "incorrect": the prediction is not acceptable.
- "needs_review": only if the evidence is genuinely ambiguous.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _fallback_decision(
    *,
    missing_fields: List[str],
    output_invalid: bool,
    output_invalid_reasons: List[str],
    rule_score: Optional[float],
    llm_error: Optional[str],
    parse_error: Optional[str],
    dry_run: bool,
) -> Dict[str, Any]:
    if missing_fields:
        reason = f"Missing required fields for judging: {', '.join(missing_fields)}"
        return {
            "verdict": "needs_review",
            "confidence": 0.0,
            "reason": reason,
            "key_points": [reason],
            "matches_ground_truth": None,
            "requested_change_applied": None,
        }

    if output_invalid:
        reason = "Prediction is not a valid full-file apply output: " + "; ".join(output_invalid_reasons[:3])
        return {
            "verdict": "incorrect",
            "confidence": 1.0,
            "reason": reason,
            "key_points": output_invalid_reasons[:3],
            "matches_ground_truth": False,
            "requested_change_applied": None,
        }

    if rule_score is not None and rule_score >= 1.0:
        reason = "Fallback accepted the prediction because rule_score=1.0 after stripping <update_file>."
        return {
            "verdict": "correct",
            "confidence": 0.95 if not dry_run else 0.8,
            "reason": reason,
            "key_points": [reason],
            "matches_ground_truth": True,
            "requested_change_applied": True,
        }

    if dry_run:
        reason = "Dry-run mode: LLM call skipped, so this sample is marked needs_review unless deterministic fallback applies."
        return {
            "verdict": "needs_review",
            "confidence": 0.0,
            "reason": reason,
            "key_points": [reason],
            "matches_ground_truth": None,
            "requested_change_applied": None,
        }

    reason = llm_error or parse_error or "LLM judgment unavailable."
    return {
        "verdict": "needs_review",
        "confidence": 0.2,
        "reason": reason,
        "key_points": [reason[:300]],
        "matches_ground_truth": None,
        "requested_change_applied": None,
    }


def _judge_record(
    record: Dict[str, Any],
    *,
    sample: ApplySample,
    config: LLMConfig,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    judge_start = time.time()

    missing_fields = [
        name
        for name in ("original_code", "ground_truth", "prediction")
        if sample.extracted_fields.get(name) is None
    ]

    output_invalid, output_invalid_reasons, parsed = check_output_invalid(
        sample.prediction_raw,
        sample.language or None,
    )
    prediction_extracted = parsed.extracted

    rule_score: Optional[float] = None
    rule_score_error: Optional[str] = None
    try:
        rule_score = float(compute_score_pygments(sample.prediction_raw, sample.ground_truth, sample.language))
    except Exception as exc:
        rule_score_error = f"rule_score_error: {exc}"

    expected_diff = unified_diff_text(
        sample.original_code,
        sample.ground_truth,
        fromfile="original",
        tofile="ground_truth",
        context_lines=args.diff_context_lines,
        max_lines=args.diff_max_lines,
    )
    actual_diff = unified_diff_text(
        sample.original_code,
        prediction_extracted,
        fromfile="original",
        tofile="prediction",
        context_lines=args.diff_context_lines,
        max_lines=args.diff_max_lines,
    )
    prediction_vs_ground_truth_diff = unified_diff_text(
        sample.ground_truth,
        prediction_extracted,
        fromfile="ground_truth",
        tofile="prediction",
        context_lines=args.diff_context_lines,
        max_lines=args.diff_max_lines,
    )

    feature_summary: Dict[str, Any] = {}
    feature_summary_error: Optional[str] = None
    try:
        feature_summary = compute_diff_feature_summary(
            original_text=sample.original_code,
            ground_truth_text=sample.ground_truth,
            prediction_text=prediction_extracted,
        )
    except Exception as exc:
        feature_summary_error = f"feature_summary_error: {exc}"

    compact_feature_summary = _build_compact_feature_summary(feature_summary)
    auto_signals = {
        "rule_score": rule_score,
        "rule_score_error": rule_score_error,
        "output_invalid": output_invalid,
        "output_invalid_reasons": output_invalid_reasons[:5],
        "prediction_wrapper": {
            "has_start_tag": parsed.has_start_tag,
            "has_end_tag": parsed.has_end_tag,
            "start_tag_count": parsed.start_tag_count,
            "end_tag_count": parsed.end_tag_count,
            "outside_text_nonempty": parsed.outside_text_nonempty,
        },
        "missing_fields": missing_fields,
        "original_code_length": len(sample.original_code),
        "ground_truth_length": len(sample.ground_truth),
        "prediction_extracted_length": len(prediction_extracted),
        "feature_summary": compact_feature_summary,
        "feature_summary_error": feature_summary_error,
    }

    used_llm = False
    raw_response: Optional[str] = None
    llm_error: Optional[str] = None
    parse_error: Optional[str] = None

    if args.dry_run:
        normalized = _fallback_decision(
            missing_fields=missing_fields,
            output_invalid=output_invalid,
            output_invalid_reasons=output_invalid_reasons,
            rule_score=rule_score,
            llm_error=None,
            parse_error=None,
            dry_run=True,
        )
    elif missing_fields:
        normalized = _fallback_decision(
            missing_fields=missing_fields,
            output_invalid=output_invalid,
            output_invalid_reasons=output_invalid_reasons,
            rule_score=rule_score,
            llm_error=None,
            parse_error=None,
            dry_run=False,
        )
    else:
        used_llm = True
        messages = _build_messages(
            sample=sample,
            expected_diff=expected_diff,
            actual_diff=actual_diff,
            prediction_vs_ground_truth_diff=prediction_vs_ground_truth_diff,
            auto_signals=auto_signals,
            args=args,
        )
        raw_response, llm_error = call_llm_chat(config, messages)

        normalized = None
        if raw_response and not llm_error:
            normalized, parse_error = _normalize_judge_output(raw_response)

        if normalized is None:
            normalized = _fallback_decision(
                missing_fields=missing_fields,
                output_invalid=output_invalid,
                output_invalid_reasons=output_invalid_reasons,
                rule_score=rule_score,
                llm_error=llm_error,
                parse_error=parse_error,
                dry_run=False,
            )

    judged = dict(record)
    judged.update(
        {
            "judge_input_file": sample.source_file,
            "judge_input_line": sample.line_no,
            "judge_result": normalized["verdict"],
            "judge_confidence": normalized["confidence"],
            "judge_reason": normalized["reason"],
            "judge_key_points": normalized["key_points"],
            "judge_matches_ground_truth": normalized["matches_ground_truth"],
            "judge_requested_change_applied": normalized["requested_change_applied"],
            "judge_model": config.model,
            "judge_elapsed_time_s": round(time.time() - judge_start, 6),
            "judge_used_llm": used_llm,
            "judge_error": llm_error or parse_error,
            "judge_output_invalid": output_invalid,
            "judge_output_invalid_reasons": output_invalid_reasons[:5],
            "judge_rule_score": rule_score,
            "judge_signals": auto_signals,
            "judge_extracted_field_names": sample.extracted_fields,
        }
    )

    if args.debug:
        judged["judge_raw_response"] = raw_response
        judged["judge_prediction_extracted"] = prediction_extracted
        judged["judge_expected_diff"] = expected_diff
        judged["judge_actual_diff"] = actual_diff
        judged["judge_prediction_vs_ground_truth_diff"] = prediction_vs_ground_truth_diff

    return judged


def _discover_input_files(input_dir: Path, pattern: str, max_files: int) -> List[Path]:
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if max_files > 0:
        files = files[:max_files]
    return files


def _ensure_writable_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    input_files = _discover_input_files(input_dir, args.input_pattern, args.max_files)
    if not input_files:
        raise FileNotFoundError(f"No jsonl files matched {args.input_pattern!r} under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    overall_summary_path = output_dir / "overall_summary.json"
    _ensure_writable_output(overall_summary_path, args.overwrite)

    llm_config = LLMConfig(
        enabled=not args.dry_run,
        model=args.llm_model,
        api_key=args.llm_api_key or None,
        base_url=args.llm_base_url,
        timeout_s=args.llm_timeout_s,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        extra_body=_parse_extra_body(args.llm_extra_body),
    )

    total_processed = 0
    overall_verdict_counter: Counter[str] = Counter()
    overall_error_counter = 0
    overall_invalid_input_counter = 0
    file_summaries: List[Dict[str, Any]] = []

    for input_file in input_files:
        judged_path = output_dir / f"{input_file.stem}.judged.jsonl"
        summary_path = output_dir / f"{input_file.stem}.summary.json"
        _ensure_writable_output(judged_path, args.overwrite)
        _ensure_writable_output(summary_path, args.overwrite)

        file_verdict_counter: Counter[str] = Counter()
        file_error_counter = 0
        file_invalid_input_counter = 0
        file_total = 0

        print(f"[INFO] judging {input_file}", flush=True)
        with judged_path.open("w", encoding="utf-8") as out_f:
            for line_no, record, input_error in _iter_jsonl(input_file):
                if args.max_samples > 0 and total_processed >= args.max_samples:
                    break

                if input_error:
                    judged = {
                        "judge_input_file": str(input_file),
                        "judge_input_line": line_no,
                        "judge_result": "needs_review",
                        "judge_confidence": 0.0,
                        "judge_reason": input_error,
                        "judge_key_points": [input_error],
                        "judge_matches_ground_truth": None,
                        "judge_requested_change_applied": None,
                        "judge_model": llm_config.model,
                        "judge_elapsed_time_s": 0.0,
                        "judge_used_llm": False,
                        "judge_error": input_error,
                        "judge_output_invalid": None,
                        "judge_output_invalid_reasons": [],
                        "judge_rule_score": None,
                        "judge_signals": {},
                        "judge_extracted_field_names": {},
                    }
                    file_invalid_input_counter += 1
                    overall_invalid_input_counter += 1
                else:
                    sample = _build_sample(input_file, line_no, record or {})
                    judged = _judge_record(record or {}, sample=sample, config=llm_config, args=args)

                out_f.write(json.dumps(judged, ensure_ascii=False) + "\n")
                file_total += 1
                total_processed += 1

                verdict = str(judged.get("judge_result") or "needs_review")
                file_verdict_counter[verdict] += 1
                overall_verdict_counter[verdict] += 1
                if judged.get("judge_error"):
                    file_error_counter += 1
                    overall_error_counter += 1

                if file_total % 20 == 0:
                    print(
                        f"[INFO] {input_file.name}: processed={file_total}, "
                        f"correct={file_verdict_counter.get('correct', 0)}, "
                        f"incorrect={file_verdict_counter.get('incorrect', 0)}, "
                        f"needs_review={file_verdict_counter.get('needs_review', 0)}",
                        flush=True,
                    )

            out_f.flush()

        file_summary = {
            "input_file": str(input_file),
            "output_file": str(judged_path),
            "model": llm_config.model,
            "dry_run": bool(args.dry_run),
            "total_records": file_total,
            "verdict_counts": dict(file_verdict_counter),
            "judge_error_count": file_error_counter,
            "invalid_input_count": file_invalid_input_counter,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(summary_path, file_summary)
        file_summaries.append(file_summary)
        print(f"[OK] wrote {judged_path}", flush=True)
        print(f"[OK] wrote {summary_path}", flush=True)

        if args.max_samples > 0 and total_processed >= args.max_samples:
            break

    overall_summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_pattern": args.input_pattern,
        "matched_input_files": [str(path) for path in input_files],
        "model": llm_config.model,
        "dry_run": bool(args.dry_run),
        "total_processed": total_processed,
        "overall_verdict_counts": dict(overall_verdict_counter),
        "overall_judge_error_count": overall_error_counter,
        "overall_invalid_input_count": overall_invalid_input_counter,
        "files": file_summaries,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(overall_summary_path, overall_summary)
    print(f"[OK] wrote {overall_summary_path}", flush=True)


if __name__ == "__main__":
    main()
