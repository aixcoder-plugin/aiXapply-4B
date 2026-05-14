from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from experiments.evaluation.diff_features import compute_diff_feature_summary, unified_diff_text
from experiments.evaluation.error_types import ErrorType
from experiments.evaluation.llm_judge import LLMConfig, classify_error_type_with_llm
from experiments.evaluation.update_file_parser import check_output_invalid


@dataclass
class ClassifierOptions:
    llm: LLMConfig
    debug: bool = False
    expected_diff_max_lines: int = 800
    actual_diff_max_lines: int = 800
    diff_context_lines: int = 2
    # Only skip LLM when the rule-side confidence reaches this threshold.
    rule_skip_llm_conf_threshold: float = 0.9


def _rule_classify(
    feature_summary: Dict[str, Any], expected_diff_nonempty: bool
) -> Tuple[ErrorType, float, List[str], bool]:
    """
    Rule-side "candidate classification":
    - Only a few highly deterministic cases get high confidence (>=0.9). Only those cases are allowed to skip LLM.
    - All other cases are assigned low confidence (<=0.6) and treated as semantically ambiguous, so LLM is recommended.

    Returns: (rule_type, rule_confidence, rule_reasons, is_deterministic_high_conf)
    """
    no_op = float(feature_summary.get("no_op_score") or 0.0)
    patch_match = float(feature_summary.get("patch_match_score") or 0.0)
    side_fx = int(feature_summary.get("side_effect_opcode_count") or 0)
    actual_edit_count = int(feature_summary.get("actual_edit_count") or 0)
    misplaced = feature_summary.get("misplaced_expected_edit_ids") or []
    missing = feature_summary.get("missing_expected_edit_ids") or []
    near_miss = feature_summary.get("near_miss_expected_edit_ids") or []

    reasons: List[str] = []

    # 1) expected_diff is empty: nothing should change; any change is a side effect (highly deterministic).
    if not expected_diff_nonempty:
        return (
            ErrorType.OUT_OF_PATCH_SIDE_EFFECT,
            0.99,
            ["expected_diff empty but prediction changed original"],
            True,
        )

    # 2) Pure no-op: prediction is essentially identical to original and has no edit opcodes (deterministic).
    if actual_edit_count == 0 and no_op >= 0.9995:
        reasons.append(f"no_op_score={no_op:.4f} and actual_edit_count=0")
        return ErrorType.PATCH_NOT_APPLIED, 0.99, reasons, True

    # 3) Patch is almost correct but there are edits outside patch windows: side effect (usually deterministic).
    if patch_match >= 0.99 and side_fx > 0:
        reasons.append(f"patch_match_score={patch_match:.3f} and side_effect_opcode_count={side_fx}")
        return ErrorType.OUT_OF_PATCH_SIDE_EFFECT, 0.92, reasons, True

    # ===== All other cases: semantically ambiguous, low confidence =====
    # Misplacement signals exist but are not always 90%+ correct, so keep low confidence and prefer LLM.
    if misplaced:
        reasons.append(f"misplaced_expected_edit_ids={misplaced[:5]}")
        return ErrorType.WRONG_POSITION, 0.6, reasons, False

    # Incomplete vs incorrect: hard to distinguish robustly via rules, keep it conservative and low-confidence.
    if missing:
        reasons.append(f"missing_expected_edit_ids={missing[:5]}")
        return ErrorType.PATCH_INCOMPLETE, 0.55, reasons, False

    if near_miss:
        reasons.append(f"near_miss_expected_edit_ids={near_miss[:5]}")
    reasons.append(f"patch_match_score={patch_match:.3f}")
    return ErrorType.PATCH_INCORRECT, 0.55, reasons, False


def _consistency_guard(
    llm_error_type: ErrorType, feature_summary: Dict[str, Any], expected_diff_nonempty: bool
) -> Tuple[ErrorType, bool, Optional[str]]:
    """
    Lightweight consistency checks on the LLM output.
    Returns: (final_type, needs_review, override_reason)
    """
    side_fx = int(feature_summary.get("side_effect_opcode_count") or 0)
    patch_match = float(feature_summary.get("patch_match_score") or 0.0)
    no_op = float(feature_summary.get("no_op_score") or 0.0)
    misplaced = feature_summary.get("misplaced_expected_edit_ids") or []

    # If expected_diff is empty, NotApplied/Incomplete/Incorrect/WrongPosition are not reasonable.
    if not expected_diff_nonempty:
        if llm_error_type != ErrorType.OUT_OF_PATCH_SIDE_EFFECT:
            return ErrorType.OUT_OF_PATCH_SIDE_EFFECT, True, "expected_diff empty; treat as side effect"

    if llm_error_type == ErrorType.OUT_OF_PATCH_SIDE_EFFECT and side_fx == 0:
        return llm_error_type, True, "LLM says side effect but rule detected no side-effect opcodes"

    if llm_error_type == ErrorType.PATCH_NOT_APPLIED:
        # If LLM says no-op, but we see out-of-patch edits or strong patch evidence, mark as review.
        if side_fx > 0:
            return llm_error_type, True, "LLM says not applied but there are out-of-patch edits"
        if no_op < 0.98 and patch_match > 0.3:
            return llm_error_type, True, "LLM says not applied but patch_match_score suggests changes"

    if llm_error_type == ErrorType.WRONG_POSITION and not misplaced:
        return llm_error_type, True, "LLM says wrong position but misplaced signals are weak"

    return llm_error_type, False, None


def classify_record(
    record: Dict[str, Any], options: ClassifierOptions
) -> Dict[str, Any]:
    """
    record must contain:
    - prediction (raw)
    - ground_truth (full file)
    - original_code (full file)
    - language (optional)
    - update_snippet (optional)
    """
    language = record.get("language")
    prediction_raw = record.get("prediction") or ""
    ground_truth = record.get("ground_truth") or ""
    original_code = record.get("original_code") or ""


    is_invalid, invalid_reasons, parsed = check_output_invalid(prediction_raw, language)
    if is_invalid:
        return {
            "error_type": ErrorType.OUTPUT_INVALID.value,
            "confidence": 1.0,
            "needs_review": False,
            "reason_brief": invalid_reasons[:3],
            "rule_error_type": ErrorType.OUTPUT_INVALID.value,
            "rule_confidence": 1.0,
            "rule_reason_brief": invalid_reasons[:3],
            "rule_deterministic": True,
            "rule_can_skip_llm": True,
            "used_llm": False,
            "llm_error_type": None,
            "llm_confidence": None,
            "llm_reason_brief": None,
            "debug": {
                "outside_text_excerpt": parsed.outside_text_excerpt,
                "start_tag_count": parsed.start_tag_count,
                "end_tag_count": parsed.end_tag_count,
            }
            if options.debug
            else None,
        }

    prediction_extracted = parsed.extracted

    # Stage B/C/D: diff features
    expected_diff = unified_diff_text(
        original_code,
        ground_truth,
        fromfile="original",
        tofile="ground_truth",
        context_lines=options.diff_context_lines,
        max_lines=options.expected_diff_max_lines,
    )
    actual_diff = unified_diff_text(
        original_code,
        prediction_extracted,
        fromfile="original",
        tofile="prediction",
        context_lines=options.diff_context_lines,
        max_lines=options.actual_diff_max_lines,
    )
    expected_diff_nonempty = expected_diff.strip() != "" and "\n" in expected_diff

    feature_summary = compute_diff_feature_summary(
        original_text=original_code,
        ground_truth_text=ground_truth,
        prediction_text=prediction_extracted,
    )

    # Rule-side candidate: always compute one (for comparison / persistence)
    rule_type, rule_conf, rule_reasons, rule_deterministic = _rule_classify(
        feature_summary, expected_diff_nonempty
    )
    # Only high-confidence rules are allowed to skip LLM.
    rule_can_skip_llm = rule_deterministic and (rule_conf >= options.rule_skip_llm_conf_threshold)

    # Stage E: LLM judging (except OutputInvalid)
    llm_result = None
    llm_err = None
    used_llm = False
    if options.llm.enabled and (not rule_can_skip_llm):
        used_llm = True
        llm_result, llm_err = classify_error_type_with_llm(
            config=options.llm,
            language=language,
            expected_diff=expected_diff,
            actual_diff=actual_diff,
            feature_summary=feature_summary,
        )

    if llm_result:
        llm_type = ErrorType(llm_result["error_type"])
        final_type, needs_review, override_reason = _consistency_guard(
            llm_type, feature_summary, expected_diff_nonempty
        )
        final_conf = float(llm_result.get("confidence", 0.5))
        final_reasons = llm_result.get("reason_brief", [])[:3]
        if override_reason:
            final_reasons = (final_reasons + [override_reason])[:3]

        out: Dict[str, Any] = {
            # Final classification (usually from LLM)
            "error_type": final_type.value,
            "confidence": final_conf,
            "needs_review": needs_review,
            "reason_brief": final_reasons,
            # Rule-side output (for comparison)
            "rule_error_type": rule_type.value,
            "rule_confidence": rule_conf,
            "rule_reason_brief": rule_reasons[:3],
            "rule_deterministic": rule_deterministic,
            "rule_can_skip_llm": rule_can_skip_llm,
            # LLM-side output (for comparison)
            "used_llm": used_llm,
            "llm_error_type": llm_type.value,
            "llm_confidence": float(llm_result.get("confidence", 0.5)),
            "llm_reason_brief": llm_result.get("reason_brief", [])[:3],
        }

        if options.debug:
            out["debug"] = {
                "llm_error": llm_err,
                "feature_summary": feature_summary,
                "expected_diff": expected_diff,
                "actual_diff": actual_diff,
            }
        return out

    # LLM not used/unavailable: fall back to the rule-side candidate
    fb_type, fb_conf, fb_reasons = rule_type, rule_conf, rule_reasons
    out2: Dict[str, Any] = {
        "error_type": fb_type.value,
        "confidence": fb_conf,
        # Mark review when the rule is not high-confidence (or LLM should have been used but failed).
        "needs_review": (not rule_can_skip_llm),
        "reason_brief": fb_reasons[:3],
        "rule_error_type": rule_type.value,
        "rule_confidence": rule_conf,
        "rule_reason_brief": rule_reasons[:3],
        "rule_deterministic": rule_deterministic,
        "rule_can_skip_llm": rule_can_skip_llm,
        "used_llm": used_llm,
        "llm_error_type": None,
        "llm_confidence": None,
        "llm_reason_brief": None,
    }
    if options.debug:
        out2["debug"] = {
            "llm_error": llm_err,
            "feature_summary": feature_summary,
            "expected_diff": expected_diff,
            "actual_diff": actual_diff,
        }
    return out2

