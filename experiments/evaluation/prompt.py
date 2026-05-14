from __future__ import annotations

from experiments.evaluation.error_types import ERROR_TYPE_RUBRIC_EN, ErrorType


ERROR_CLASSIFIER_SYSTEM_PROMPT = """You are a strict classification engine for patch-application errors.
You must output ONLY valid JSON (no markdown, no extra text).
"""


def build_error_classifier_user_prompt(
    *,
    language: str | None,
    expected_diff: str,
    actual_diff: str,
    feature_summary_json: str,
) -> str:
    rubric_lines = []
    for k in ErrorType:
        rubric_lines.append(f"- {k.value}: {ERROR_TYPE_RUBRIC_EN[k]}")
    rubric_text = "\n".join(rubric_lines)

    return f"""Task: Classify the error type of an apply-model output that is already known to be WRONG (prediction != ground truth).

Choose exactly ONE error_type from the allowed set below, based on expected vs actual diffs and the provided feature summary.

Allowed error types and rubric:
{rubric_text}

Input language: {language or "unknown"}

expected_diff (original -> ground_truth):
<<<EXPECTED_DIFF
{expected_diff}
EXPECTED_DIFF

actual_diff (original -> prediction):
<<<ACTUAL_DIFF
{actual_diff}
ACTUAL_DIFF

feature_summary (JSON):
<<<FEATURES
{feature_summary_json}
FEATURES

Important notes:
- PATCH_INCOMPLETE is ONLY when at least one full expected line/block is truly ABSENT from the prediction.
- If the expected change exists but differs by whitespace or small typos (e.g., trailing spaces), classify as PATCH_INCORRECT (not PATCH_INCOMPLETE).
- If feature_summary contains "near_miss_expected_edit_ids", those edits are "attempted but not exactly matched" (often whitespace mismatch) and should be treated as PATCH_INCORRECT unless there are other clearly missing edits.

Output requirements:
- Output ONLY a JSON object (no markdown).
- JSON schema:
  {{
    "error_type": "<one of the allowed error types>",
    "confidence": <number between 0 and 1>,
    "reason_brief": [<1-3 short bullet reasons, must cite lines/snippets from diffs or features>]
  }}
"""

