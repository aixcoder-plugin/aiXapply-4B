from __future__ import annotations

from enum import Enum


class ErrorType(str, Enum):
    OUTPUT_INVALID = "OUTPUT_INVALID"
    PATCH_NOT_APPLIED = "PATCH_NOT_APPLIED"
    PATCH_INCOMPLETE = "PATCH_INCOMPLETE"
    PATCH_INCORRECT = "PATCH_INCORRECT"
    WRONG_POSITION = "WRONG_POSITION"
    OUT_OF_PATCH_SIDE_EFFECT = "OUT_OF_PATCH_SIDE_EFFECT"


ERROR_TYPE_RUBRIC_EN = {
    ErrorType.OUTPUT_INVALID: (
        "Output cannot be used as a full-file result: missing/wrong <update_file> wrapper, "
        "truncated/incomplete output, placeholder leakage (e.g. '... existing code ...'), "
        "mixed narrative text, or invalid structured format (JSON/YAML/XML/INI cannot be parsed)."
    ),
    ErrorType.PATCH_NOT_APPLIED: (
        "Expected changes (expected_diff) are largely absent; output remains like original in patch area."
    ),
    ErrorType.PATCH_INCOMPLETE: (
        "Some expected changes are applied but at least one full expected line/block is missing."
    ),
    ErrorType.PATCH_INCORRECT: (
        "Patch area is edited but does not match expected change; wrong tokens/strings/logic compared to GT/snippet."
    ),
    ErrorType.WRONG_POSITION: (
        "Correct new content exists but placed in wrong location, or original target location not correctly updated."
    ),
    ErrorType.OUT_OF_PATCH_SIDE_EFFECT: (
        "Patch is correct (or mostly correct) but there are modifications outside expected patch region."
    ),
}

