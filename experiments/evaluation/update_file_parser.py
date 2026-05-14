from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


try:
    import yaml  # type: ignore

    _YAML_AVAILABLE = True
except Exception:
    _YAML_AVAILABLE = False

try:
    from xml.etree import ElementTree as ET

    _XML_AVAILABLE = True
except Exception:
    _XML_AVAILABLE = False

try:
    from configparser import ConfigParser

    _INI_AVAILABLE = True
except Exception:
    _INI_AVAILABLE = False


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


_PLACEHOLDER_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*(//|#|;|--)?\s*\.\.\.\s*existing\s*code\s*\.\.\.\s*$", re.IGNORECASE),
    re.compile(r"^\s*(//|#|;|--)?\s*\.\.\.\s*more\s*code\s*\.\.\.\s*$", re.IGNORECASE),
    re.compile(r"^\s*(//|#|;|--)?\s*\.\.\.\s*$"),
]


def detect_placeholder_leakage(text: str) -> Tuple[bool, List[str]]:
    """
    Detect typical placeholder leakage in the output (primarily line-level matching to avoid
    false positives on string literals).
    Returns: (has_leakage, matched_lines_sample)
    """
    matches: List[str] = []
    for line in normalize_line_endings(text).splitlines():
        for pat in _PLACEHOLDER_PATTERNS:
            if pat.match(line):
                matches.append(line.strip())
                if len(matches) >= 5:
                    return True, matches
                break
    return (len(matches) > 0), matches


@dataclass(frozen=True)
class ParsedUpdateFile:
    raw: str
    extracted: str
    has_start_tag: bool
    has_end_tag: bool
    start_tag_count: int
    end_tag_count: int
    outside_text_nonempty: bool
    outside_text_excerpt: str


def extract_update_file_block(prediction_raw: str) -> ParsedUpdateFile:
    raw = prediction_raw or ""
    raw = normalize_line_endings(raw)

    start_tag = "<update_file>"
    end_tag = "</update_file>"

    start_count = raw.count(start_tag)
    end_count = raw.count(end_tag)

    start_idx = raw.find(start_tag)
    end_idx = raw.rfind(end_tag)

    has_start = start_idx != -1
    has_end = end_idx != -1 and end_idx > start_idx

    extracted = ""
    outside_nonempty = False
    outside_excerpt = ""

    if has_start and has_end:
        extracted = raw[start_idx + len(start_tag) : end_idx]
        prefix = raw[:start_idx]
        suffix = raw[end_idx + len(end_tag) :]
        outside_text = (prefix + "\n" + suffix).strip()
        outside_nonempty = bool(outside_text)
        if outside_nonempty:
            outside_excerpt = outside_text[:300]
    elif has_start and not has_end:
        # Truncated: start tag exists but end tag is missing
        extracted = raw[start_idx + len(start_tag) :]
        outside_text = raw[:start_idx].strip()
        outside_nonempty = bool(outside_text)
        outside_excerpt = outside_text[:300] if outside_nonempty else ""
    else:
        # No wrapper: keep raw text for further checks
        extracted = raw
        outside_nonempty = False
        outside_excerpt = ""

    return ParsedUpdateFile(
        raw=raw,
        extracted=extracted,
        has_start_tag=has_start,
        has_end_tag=has_end,
        start_tag_count=start_count,
        end_tag_count=end_count,
        outside_text_nonempty=outside_nonempty,
        outside_text_excerpt=outside_excerpt,
    )


def _is_parsable_structured_format(text: str, language: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Parse-check structured formats (only used for OUTPUT_INVALID).
    Returns: (ok, error_message)
    """
    if not language:
        return True, None
    lang = language.lower().strip()
    payload = text.strip()
    if not payload:
        return False, "empty_content"

    if lang == "json":
        try:
            json.loads(payload)
            return True, None
        except Exception as e:
            return False, f"json_parse_error: {e}"

    if lang in ("yaml", "yml"):
        if not _YAML_AVAILABLE:
            return True, None
        try:
            list(yaml.safe_load_all(payload))  # type: ignore[name-defined]
            return True, None
        except Exception as e:
            return False, f"yaml_parse_error: {e}"

    if lang == "xml":
        if not _XML_AVAILABLE:
            return True, None
        try:
            ET.fromstring(payload)  # type: ignore[name-defined]
            return True, None
        except Exception as e:
            return False, f"xml_parse_error: {e}"

    if lang == "ini":
        if not _INI_AVAILABLE:
            return True, None
        try:
            parser = ConfigParser()  # type: ignore[name-defined]
            parser.read_string(payload)
            return True, None
        except Exception as e:
            return False, f"ini_parse_error: {e}"

    return True, None


def check_output_invalid(prediction_raw: str, language: Optional[str] = None) -> Tuple[bool, List[str], ParsedUpdateFile]:
    """
    Rule-side hard-gate: only handles deterministic OUTPUT_INVALID checks.
    Returns: (is_invalid, reasons, parsed)
    """
    parsed = extract_update_file_block(prediction_raw)
    reasons: List[str] = []

    # 1) Wrapper missing / not paired / multiple
    if not parsed.has_start_tag or not parsed.has_end_tag:
        reasons.append("wrapper_missing_or_truncated")
    if parsed.start_tag_count != 1 or parsed.end_tag_count != 1:
        # Multiple wrappers or unpaired tags
        reasons.append("wrapper_count_abnormal")

    # 2) Non-whitespace text outside wrapper
    if parsed.outside_text_nonempty:
        reasons.append("extra_text_outside_wrapper")

    # 3) Placeholder leakage
    has_ph, ph_lines = detect_placeholder_leakage(parsed.extracted)
    if has_ph:
        reasons.append("placeholder_leakage")
        # Only attach a small sample to avoid bloating reasons
        reasons.extend([f"placeholder: {line}" for line in ph_lines[:3]])

    # 4) Structured format cannot be parsed
    ok, err = _is_parsable_structured_format(parsed.extracted, language)
    if not ok:
        reasons.append("invalid_structured_format")
        if err:
            reasons.append(err)

    return (len(reasons) > 0), reasons, parsed

