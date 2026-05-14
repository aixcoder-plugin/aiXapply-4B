from __future__ import annotations

import math
from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from typing import Dict, List, Optional, Sequence, Tuple

from experiments.evaluation.update_file_parser import normalize_line_endings


def _to_lines(text: str) -> List[str]:
    return normalize_line_endings(text or "").splitlines()


def _to_lines_keepends(text: str) -> List[str]:
    return normalize_line_endings(text or "").splitlines(keepends=True)


def _safe_ratio(a: Sequence, b: Sequence) -> float:
    try:
        return SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0


def unified_diff_text(
    a_text: str,
    b_text: str,
    fromfile: str,
    tofile: str,
    context_lines: int = 2,
    max_lines: Optional[int] = 800,
) -> str:
    # NOTE: Use splitlines() WITHOUT keepends, then join with "\n".
    # Otherwise unified_diff() will keep original "\n" in each line, and join will double blank lines.
    a_lines = _to_lines(a_text)
    b_lines = _to_lines(b_text)
    diff_lines = list(
        unified_diff(
            a_lines,
            b_lines,
            fromfile=fromfile,
            tofile=tofile,
            n=context_lines,
            lineterm="",
        )
    )
    if max_lines is not None and len(diff_lines) > max_lines:
        head = diff_lines[: max_lines // 2]
        tail = diff_lines[-max_lines // 2 :]
        marker = [f"... (diff truncated, total_lines={len(diff_lines)})"]
        diff_lines = head + marker + tail
    return "\n".join(diff_lines)


def _is_informative_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if len(s) <= 3:
        return False
    # Avoid treating pure punctuation/brackets as a signature line.
    only_punct = all(ch in "{}[]();,:+-*/=<>|&!." for ch in s)
    if only_punct:
        return False
    # Avoid treating placeholder leakage as a signature line.
    lower = s.lower()
    if "existing code" in lower or "more code" in lower:
        return False
    return True


@dataclass(frozen=True)
class ExpectedEdit:
    edit_id: int
    tag: str  # insert/replace/delete
    orig_range: Tuple[int, int]  # [i1, i2)
    old_lines: List[str]
    new_lines: List[str]
    anchor_before: List[str]
    anchor_after: List[str]
    signature_lines: List[str]


@dataclass(frozen=True)
class ExpectedEditMatch:
    edit_id: int
    tag: str
    expected_new_len: int
    anchor_pred_idx: int
    window: Tuple[int, int]
    local_ratio: float
    local_coverage: float
    global_coverage: float
    signature_found_anywhere: bool
    signature_found_near_anchor: bool
    nearest_signature_distance: Optional[int]


def build_expected_edits(
    original_text: str, ground_truth_text: str, context: int = 3, signature_max: int = 5
) -> List[ExpectedEdit]:
    orig_lines = _to_lines(original_text)
    gt_lines = _to_lines(ground_truth_text)
    sm = SequenceMatcher(None, orig_lines, gt_lines)

    edits: List[ExpectedEdit] = []
    next_id = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old = orig_lines[i1:i2]
        new = gt_lines[j1:j2]
        before = orig_lines[max(0, i1 - context) : i1]
        after = orig_lines[i2 : min(len(orig_lines), i2 + context)]
        sig: List[str] = []
        for line in new:
            if _is_informative_line(line):
                s = line.strip()
                if s not in sig:
                    sig.append(s)
                if len(sig) >= signature_max:
                    break
        edits.append(
            ExpectedEdit(
                edit_id=next_id,
                tag=tag,
                orig_range=(i1, i2),
                old_lines=old,
                new_lines=new,
                anchor_before=before,
                anchor_after=after,
                signature_lines=sig,
            )
        )
        next_id += 1
    return edits


def _build_expected_patch_windows(
    edits: List[ExpectedEdit], original_len: int, buffer: int = 3
) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    for e in edits:
        i1, i2 = e.orig_range
        # For insert (i1 == i2), still expand a small window around the point.
        start = max(0, i1 - buffer)
        end = min(original_len, i2 + buffer)
        windows.append((start, end))
    # merge overlaps
    windows.sort()
    merged: List[Tuple[int, int]] = []
    for w in windows:
        if not merged:
            merged.append(w)
            continue
        ps, pe = merged[-1]
        cs, ce = w
        if cs <= pe:
            merged[-1] = (ps, max(pe, ce))
        else:
            merged.append(w)
    return merged


def _intersects_point_or_range(
    i1: int, i2: int, windows: List[Tuple[int, int]]
) -> bool:
    # insert: i1==i2, treat as point
    if i1 == i2:
        p = i1
        for s, e in windows:
            if s <= p <= e:
                return True
        return False
    for s, e in windows:
        if i2 <= s:
            continue
        if i1 >= e:
            continue
        return True
    return False


def _map_orig_index_to_pred_index(
    opcodes: List[Tuple[str, int, int, int, int]], orig_idx: int
) -> int:
    """
    Build a rough mapping from original -> prediction using opcodes.
    Given an original line index, return an approximate line index in the prediction.
    """
    last_j = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if orig_idx < i1:
            break
        last_j = j2
        if i1 <= orig_idx < i2:
            if tag == "equal":
                return j1 + (orig_idx - i1)
            return j1
        if orig_idx == i2 and tag == "insert":
            return j2
    return last_j


def _coverage(expected: List[str], actual: List[str]) -> Tuple[float, float]:
    """
    Returns (ratio, coverage):
    - ratio: overall SequenceMatcher ratio
    - coverage: fraction of expected covered by matching blocks (closer to the notion of "missing full lines")
    """
    if not expected:
        return 1.0, 1.0
    sm = SequenceMatcher(None, expected, actual)
    blocks = sm.get_matching_blocks()
    match_len = sum(b.size for b in blocks)
    cov = match_len / max(1, len(expected))
    return sm.ratio(), cov


def match_expected_edits_in_prediction(
    original_text: str,
    prediction_text: str,
    expected_edits: List[ExpectedEdit],
    window_radius: int = 30,
    misplaced_distance_threshold: int = 30,
) -> List[ExpectedEditMatch]:
    orig_lines = _to_lines(original_text)
    pred_lines = _to_lines(prediction_text)
    opcodes = SequenceMatcher(None, orig_lines, pred_lines).get_opcodes()

    matches: List[ExpectedEditMatch] = []
    for e in expected_edits:
        i1, _i2 = e.orig_range
        anchor_pred = _map_orig_index_to_pred_index(opcodes, i1)
        w_start = max(0, anchor_pred - window_radius)
        w_end = min(len(pred_lines), anchor_pred + window_radius)
        pred_window = pred_lines[w_start:w_end]

        local_ratio, local_cov = _coverage(e.new_lines, pred_window)
        global_ratio, global_cov = _coverage(e.new_lines, pred_lines)

        # Signature: whether it appears anywhere / near the anchor.
        sig_positions: List[int] = []
        if e.signature_lines:
            sig_set = set(s.strip() for s in e.signature_lines if s.strip())
            for idx, line in enumerate(pred_lines):
                if line.strip() in sig_set:
                    sig_positions.append(idx)
                    if len(sig_positions) >= 50:
                        break
        found_any = len(sig_positions) > 0
        found_near = any(abs(p - anchor_pred) <= window_radius for p in sig_positions)
        nearest_dist = None
        if sig_positions:
            nearest_dist = min(abs(p - anchor_pred) for p in sig_positions)
            if nearest_dist < 0:
                nearest_dist = None

        # Strong misplacement signal: signature appears but is far away from the anchor.
        if found_any and (not found_near) and nearest_dist is not None:
            if nearest_dist < misplaced_distance_threshold:
                # Close but outside the window: likely the window is too small; ignore.
                pass

        matches.append(
            ExpectedEditMatch(
                edit_id=e.edit_id,
                tag=e.tag,
                expected_new_len=len(e.new_lines),
                anchor_pred_idx=anchor_pred,
                window=(w_start, w_end),
                local_ratio=float(local_ratio),
                local_coverage=float(local_cov),
                global_coverage=float(global_cov),
                signature_found_anywhere=found_any,
                signature_found_near_anchor=found_near,
                nearest_signature_distance=nearest_dist,
            )
        )
    return matches


def compute_diff_feature_summary(
    original_text: str,
    ground_truth_text: str,
    prediction_text: str,
    context_buffer: int = 3,
) -> Dict[str, object]:
    orig_lines = _to_lines(original_text)
    gt_lines = _to_lines(ground_truth_text)
    pred_lines = _to_lines(prediction_text)

    expected_edits = build_expected_edits(original_text, ground_truth_text)
    expected_windows = _build_expected_patch_windows(expected_edits, len(orig_lines), buffer=context_buffer)

    orig_pred_sm = SequenceMatcher(None, orig_lines, pred_lines)
    orig_pred_opcodes = list(orig_pred_sm.get_opcodes())
    side_effect_opcodes = []
    patch_area_opcodes = []

    for tag, i1, i2, j1, j2 in orig_pred_opcodes:
        if tag == "equal":
            continue
        if _intersects_point_or_range(i1, i2, expected_windows):
            patch_area_opcodes.append((tag, i1, i2, j1, j2))
        else:
            side_effect_opcodes.append((tag, i1, i2, j1, j2))

    matches = match_expected_edits_in_prediction(original_text, prediction_text, expected_edits)

    # patch_match_score: length-weighted average of local_coverage by expected_new_len
    total_expected_new = sum(max(0, m.expected_new_len) for m in matches)
    weighted_cov = 0.0
    if total_expected_new > 0:
        for m in matches:
            weighted_cov += float(m.local_coverage) * float(m.expected_new_len)
        patch_match_score = weighted_cov / total_expected_new
    else:
        # No expected changes: treat patch_match_score as 1
        patch_match_score = 1.0

    # no-op score: overall similarity between original and prediction
    no_op_score = float(_safe_ratio(orig_lines, pred_lines))

    expected_edit_count = len(expected_edits)
    actual_edit_count = sum(1 for tag, *_ in orig_pred_opcodes if tag != "equal")

    # Estimate missing_expected_edit_ids (roughly):
    # - The old logic relied purely on strict line-equality global_coverage, which can misclassify
    #   whitespace-only differences (e.g., trailing spaces) as "missing full lines", leading to PATCH_INCOMPLETE.
    # - The new logic marks near-miss when strict coverage is low but the anchor window contains most expected
    #   new content after strip-normalization, which better matches PATCH_INCORRECT.
    expected_by_id = {e.edit_id: e for e in expected_edits}
    near_miss_edits: List[int] = []
    near_miss_set = set()
    for m in matches:
        if m.expected_new_len <= 0:
            continue
        # If strict coverage is already high enough, no need to mark near-miss.
        if m.local_coverage >= 0.2:
            continue
        e = expected_by_id.get(m.edit_id)
        if not e:
            continue
        w_start, w_end = m.window
        pred_window = pred_lines[w_start:w_end]
        window_strip_set = {ln.strip() for ln in pred_window if ln is not None}
        # Prefer more informative lines to reduce false positives.
        informative = [ln for ln in e.new_lines if _is_informative_line(ln)]
        check_lines = informative if informative else [ln for ln in e.new_lines if (ln or "").strip()]
        if not check_lines:
            continue
        hit = sum(1 for ln in check_lines if ln.strip() in window_strip_set)
        strip_cov = hit / max(1, len(check_lines))
        if strip_cov >= 0.8:
            near_miss_edits.append(m.edit_id)
            near_miss_set.add(m.edit_id)

    # If strict global_coverage is very low and it's not a near-miss, treat it as missing.
    missing_edits = [
        m.edit_id
        for m in matches
        if m.global_coverage < 0.2 and m.expected_new_len > 0 and m.edit_id not in near_miss_set
    ]
    misplaced_signals = [
        m.edit_id
        for m in matches
        if m.signature_found_anywhere and (not m.signature_found_near_anchor) and (m.nearest_signature_distance or 0) >= 30
    ]

    summary: Dict[str, object] = {
        "no_op_score": no_op_score,
        "patch_match_score": float(patch_match_score),
        "expected_edit_count": expected_edit_count,
        "actual_edit_count": actual_edit_count,
        "side_effect_opcode_count": len(side_effect_opcodes),
        "patch_area_opcode_count": len(patch_area_opcodes),
        "expected_patch_windows": expected_windows,
        "missing_expected_edit_ids": missing_edits,
        "near_miss_expected_edit_ids": near_miss_edits,
        "misplaced_expected_edit_ids": misplaced_signals,
        "expected_edits": [
            {
                "edit_id": e.edit_id,
                "tag": e.tag,
                "orig_range": e.orig_range,
                "signature_lines": e.signature_lines[:3],
                "old_preview": e.old_lines[:5],
                "new_preview": e.new_lines[:5],
            }
            for e in expected_edits[:50]
        ],
        "expected_edit_matches": [
            {
                "edit_id": m.edit_id,
                "tag": m.tag,
                "expected_new_len": m.expected_new_len,
                "anchor_pred_idx": m.anchor_pred_idx,
                "window": m.window,
                "local_ratio": m.local_ratio,
                "local_coverage": m.local_coverage,
                "global_coverage": m.global_coverage,
                "signature_found_anywhere": m.signature_found_anywhere,
                "signature_found_near_anchor": m.signature_found_near_anchor,
                "nearest_signature_distance": m.nearest_signature_distance,
            }
            for m in matches[:200]
        ],
        "side_effect_opcodes_preview": side_effect_opcodes[:30],
    }
    return summary

