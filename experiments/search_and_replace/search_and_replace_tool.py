"""
Python implementation of an OpenCode-like search-and-replace executor.

This module is meant for evaluation:
- Input: original_code (string), patch (JSON string / object) describing edits
- Output: patched_code (string), or raises an error if edits cannot be applied

Patch format (compatible with OpenCode Edit tool args):
Either a single object or an array of objects:
  {"oldString": "...", "newString": "...", "replaceAll": false}
  [{"oldString": "...", "newString": "...", "replaceAll": false}, ...]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple, Union


class SearchAndReplaceError(ValueError):
    """Raised when a search-and-replace patch cannot be applied."""


@dataclass(frozen=True)
class Edit:
    oldString: str
    newString: str
    replaceAll: bool = False


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n")


def _detect_line_ending(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _convert_to_line_ending(text: str, ending: str) -> str:
    if ending == "\n":
        return text
    # ending == "\r\n"
    return text.replace("\n", "\r\n")


def _levenshtein(a: str, b: str) -> int:
    if not a or not b:
        return max(len(a), len(b))
    # classic DP
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


# Similarity thresholds (mirrors OpenCode defaults)
_SINGLE_CANDIDATE_SIM_THRESHOLD = 0.0
_MULTI_CANDIDATE_SIM_THRESHOLD = 0.3


Replacer = Iterator[str]


def _simple_replacer(_content: str, find: str) -> Iterator[str]:
    yield find


def _line_trimmed_replacer(content: str, find: str) -> Iterator[str]:
    original_lines = content.split("\n")
    search_lines = find.split("\n")
    if search_lines and search_lines[-1] == "":
        search_lines.pop()

    n = len(search_lines)
    if n == 0:
        return

    for i in range(0, len(original_lines) - n + 1):
        ok = True
        for j in range(n):
            if original_lines[i + j].strip() != search_lines[j].strip():
                ok = False
                break
        if not ok:
            continue

        match_start = sum(len(original_lines[k]) + 1 for k in range(i))
        match_end = match_start
        for k in range(n):
            match_end += len(original_lines[i + k])
            if k < n - 1:
                match_end += 1
        yield content[match_start:match_end]


def _block_anchor_replacer(content: str, find: str) -> Iterator[str]:
    original_lines = content.split("\n")
    search_lines = find.split("\n")

    if len(search_lines) < 3:
        return
    if search_lines and search_lines[-1] == "":
        search_lines.pop()

    first = search_lines[0].strip()
    last = search_lines[-1].strip()
    search_block_size = len(search_lines)

    candidates: List[Tuple[int, int]] = []
    for i in range(len(original_lines)):
        if original_lines[i].strip() != first:
            continue
        for j in range(i + 2, len(original_lines)):
            if original_lines[j].strip() == last:
                candidates.append((i, j))
                break

    if not candidates:
        return

    if len(candidates) == 1:
        start_line, end_line = candidates[0]
        actual_block_size = end_line - start_line + 1
        lines_to_check = min(search_block_size - 2, actual_block_size - 2)

        similarity = 1.0 if lines_to_check <= 0 else 0.0
        if lines_to_check > 0:
            for j in range(1, min(search_block_size - 1, actual_block_size - 1)):
                o = original_lines[start_line + j].strip()
                s = search_lines[j].strip()
                max_len = max(len(o), len(s))
                if max_len == 0:
                    continue
                d = _levenshtein(o, s)
                similarity += (1 - d / max_len) / lines_to_check
                if similarity >= _SINGLE_CANDIDATE_SIM_THRESHOLD:
                    break

        if similarity >= _SINGLE_CANDIDATE_SIM_THRESHOLD:
            match_start = sum(len(original_lines[k]) + 1 for k in range(start_line))
            match_end = match_start
            for k in range(start_line, end_line + 1):
                match_end += len(original_lines[k])
                if k < end_line:
                    match_end += 1
            yield content[match_start:match_end]
        return

    # multiple candidates -> choose best similarity
    best: Tuple[int, int] | None = None
    best_sim = -1.0
    for start_line, end_line in candidates:
        actual_block_size = end_line - start_line + 1
        lines_to_check = min(search_block_size - 2, actual_block_size - 2)
        if lines_to_check <= 0:
            sim = 1.0
        else:
            sim_sum = 0.0
            for j in range(1, min(search_block_size - 1, actual_block_size - 1)):
                o = original_lines[start_line + j].strip()
                s = search_lines[j].strip()
                max_len = max(len(o), len(s))
                if max_len == 0:
                    continue
                d = _levenshtein(o, s)
                sim_sum += 1 - d / max_len
            sim = sim_sum / lines_to_check

        if sim > best_sim:
            best_sim = sim
            best = (start_line, end_line)

    if best is None or best_sim < _MULTI_CANDIDATE_SIM_THRESHOLD:
        return

    start_line, end_line = best
    match_start = sum(len(original_lines[k]) + 1 for k in range(start_line))
    match_end = match_start
    for k in range(start_line, end_line + 1):
        match_end += len(original_lines[k])
        if k < end_line:
            match_end += 1
    yield content[match_start:match_end]


def _whitespace_normalized_replacer(content: str, find: str) -> Iterator[str]:
    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    normalized_find = norm(find)
    lines = content.split("\n")

    for line in lines:
        if norm(line) == normalized_find:
            yield line
            continue
        normalized_line = norm(line)
        if normalized_find and normalized_find in normalized_line:
            words = find.strip().split()
            if not words:
                continue
            pattern = r"\s+".join(re.escape(w) for w in words)
            try:
                m = re.search(pattern, line)
            except re.error:
                m = None
            if m:
                yield m.group(0)

    find_lines = find.split("\n")
    if len(find_lines) > 1:
        for i in range(0, len(lines) - len(find_lines) + 1):
            block = "\n".join(lines[i : i + len(find_lines)])
            if norm(block) == normalized_find:
                yield block


def _indentation_flexible_replacer(content: str, find: str) -> Iterator[str]:
    def remove_indent(text: str) -> str:
        ls = text.split("\n")
        non_empty = [l for l in ls if l.strip()]
        if not non_empty:
            return text
        min_indent = min(len(re.match(r"^(\s*)", l).group(1) or "") for l in non_empty)  # type: ignore[union-attr]
        return "\n".join(l if not l.strip() else l[min_indent:] for l in ls)

    normalized_find = remove_indent(find)
    content_lines = content.split("\n")
    find_lines = find.split("\n")
    n = len(find_lines)
    if n == 0:
        return
    for i in range(0, len(content_lines) - n + 1):
        block = "\n".join(content_lines[i : i + n])
        if remove_indent(block) == normalized_find:
            yield block


def _escape_normalized_replacer(content: str, find: str) -> Iterator[str]:
    def unescape(s: str) -> str:
        def repl(m: re.Match[str]) -> str:
            c = m.group(1)
            if c == "n":
                return "\n"
            if c == "t":
                return "\t"
            if c == "r":
                return "\r"
            if c == "'":
                return "'"
            if c == '"':
                return '"'
            if c == "`":
                return "`"
            if c == "\\":
                return "\\"
            if c == "\n":
                return "\n"
            if c == "$":
                return "$"
            return m.group(0)

        return re.sub(r"\\(n|t|r|'|\"|`|\\|\n|\$)", repl, s)

    unescaped_find = unescape(find)
    if unescaped_find and unescaped_find in content:
        yield unescaped_find

    lines = content.split("\n")
    find_lines = unescaped_find.split("\n")
    n = len(find_lines)
    if n == 0:
        return
    for i in range(0, len(lines) - n + 1):
        block = "\n".join(lines[i : i + n])
        if unescape(block) == unescaped_find:
            yield block


def _trimmed_boundary_replacer(content: str, find: str) -> Iterator[str]:
    trimmed_find = find.strip()
    if trimmed_find == find:
        return
    if trimmed_find and trimmed_find in content:
        yield trimmed_find

    lines = content.split("\n")
    find_lines = find.split("\n")
    n = len(find_lines)
    if n == 0:
        return
    for i in range(0, len(lines) - n + 1):
        block = "\n".join(lines[i : i + n])
        if block.strip() == trimmed_find:
            yield block


def _context_aware_replacer(content: str, find: str) -> Iterator[str]:
    find_lines = find.split("\n")
    if len(find_lines) < 3:
        return
    if find_lines and find_lines[-1] == "":
        find_lines.pop()

    content_lines = content.split("\n")
    first = find_lines[0].strip()
    last = find_lines[-1].strip()

    for i in range(len(content_lines)):
        if content_lines[i].strip() != first:
            continue
        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() != last:
                continue
            block_lines = content_lines[i : j + 1]
            if len(block_lines) != len(find_lines):
                break
            matching = 0
            non_empty = 0
            for k in range(1, len(block_lines) - 1):
                b = block_lines[k].strip()
                f = find_lines[k].strip()
                if b or f:
                    non_empty += 1
                    if b == f:
                        matching += 1
            if non_empty == 0 or matching / non_empty >= 0.5:
                yield "\n".join(block_lines)
                break
            break


def _multi_occurrence_replacer(content: str, find: str) -> Iterator[str]:
    start = 0
    while True:
        idx = content.find(find, start)
        if idx == -1:
            break
        yield find
        start = idx + len(find)


def replace_opencode_like(content: str, oldString: str, newString: str, replaceAll: bool = False) -> str:
    """
    Apply a single edit to content, mirroring OpenCode's replace() behavior.
    """
    if oldString == newString:
        raise SearchAndReplaceError("No changes to apply: oldString and newString are identical.")

    not_found = True
    replacers = [
        _simple_replacer,
        _line_trimmed_replacer,
        _block_anchor_replacer,
        _whitespace_normalized_replacer,
        _indentation_flexible_replacer,
        _escape_normalized_replacer,
        _trimmed_boundary_replacer,
        _context_aware_replacer,
        _multi_occurrence_replacer,
    ]

    for rep in replacers:
        for search in rep(content, oldString):
            idx = content.find(search)
            if idx == -1:
                continue
            not_found = False
            if replaceAll:
                return content.replace(search, newString)
            last = content.rfind(search)
            if idx != last:
                continue
            return content[:idx] + newString + content[idx + len(search) :]

    if not_found:
        raise SearchAndReplaceError(
            "Could not find oldString in the file. It must match exactly, including whitespace, indentation, and line endings."
        )
    raise SearchAndReplaceError(
        "Found multiple matches for oldString. Provide more surrounding context to make the match unique."
    )


_FENCED_CODE_BLOCK_RE = re.compile(r"```(?P<lang>[^\n`]*)\r?\n(?P<body>.*?)(?:\r?\n)?```", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)

def _looks_like_patch_json(data: Any) -> bool:
    if isinstance(data, dict):
        if "edits" in data:
            return isinstance(data["edits"], list) and _looks_like_patch_json(data["edits"])
        return "oldString" in data and "newString" in data

    if isinstance(data, list):
        return bool(data) and all(
            isinstance(item, dict) and "oldString" in item and "newString" in item for item in data
        )

    return False

def _strip_think_blocks(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text).strip()


def _extract_fenced_blocks(text: str) -> List[str]:
    json_blocks: List[str] = []
    other_blocks: List[str] = []

    for match in _FENCED_CODE_BLOCK_RE.finditer(text):
        lang = (match.group("lang") or "").strip().lower()
        body = (match.group("body") or "").strip()
        if not body:
            continue
        if lang in {"json", "jsonc"}:
            json_blocks.append(body)
        else:
            other_blocks.append(body)

    return json_blocks + other_blocks


def _try_load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _try_extract_embedded_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    fallback: Any = None

    for idx, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            data, _ = decoder.raw_decode(text[idx:])
        except Exception:
            continue
        if _looks_like_patch_json(data):
            return data
        if fallback is None:
            fallback = data

    return fallback


def _extract_json(text: str) -> Any:
    """
    Best-effort extraction of JSON from a model output.
    - First try full-string json.loads
    - Then try fenced code blocks such as ```json ... ``` or ``` ... ```
    - Then scan the surrounding text for embedded JSON objects/arrays
    """
    text = text.strip()
    if not text:
        raise SearchAndReplaceError("Patch is empty.")

    fallback: Any = None

    def consider(data: Any) -> Any:
        nonlocal fallback
        if data is None:
            return None
        if _looks_like_patch_json(data):
            return data
        if fallback is None:
            fallback = data
        return None
    text = _strip_think_blocks(text)
    direct_candidates = [text]

    for candidate in direct_candidates:
        chosen = consider(_try_load_json(candidate))
        if chosen is not None:
            return chosen

    fenced_candidates: List[str] = []
    seen_fenced: set[str] = set()
    for source in direct_candidates:
        for block in _extract_fenced_blocks(source):
            if block in seen_fenced:
                continue
            seen_fenced.add(block)
            fenced_candidates.append(block)

    for candidate in fenced_candidates:
        chosen = consider(_try_load_json(candidate))
        if chosen is not None:
            return chosen
        chosen = consider(_try_extract_embedded_json(candidate))
        if chosen is not None:
            return chosen

    for candidate in direct_candidates:
        chosen = consider(_try_extract_embedded_json(candidate))
        if chosen is not None:
            return chosen

    if fallback is not None:
        return fallback
    raise SearchAndReplaceError("Patch is not valid JSON and no JSON object/array could be extracted.")


def parse_patch(patch: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[Edit]:
    """
    Parse patch into a list of Edit objects.
    """
    data: Any
    if isinstance(patch, str):
        data = _extract_json(patch)
    else:
        data = patch

    # allow {"edits": [...]} wrapper
    if isinstance(data, dict) and "edits" in data and isinstance(data["edits"], list):
        data = data["edits"]

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise SearchAndReplaceError(f"Unexpected patch JSON type: {type(data).__name__}")

    edits: List[Edit] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise SearchAndReplaceError(f"Patch item #{i} is not an object: {item!r}")
        if "oldString" not in item or "newString" not in item:
            raise SearchAndReplaceError(f"Patch item #{i} must contain oldString and newString.")
        edits.append(
            Edit(
                oldString=str(item["oldString"]),
                newString=str(item["newString"]),
                replaceAll=bool(item.get("replaceAll", False)),
            )
        )
    return edits


def apply_search_and_replace(original_code: str, patch: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> str:
    """
    Apply a patch (OpenCode edit-style JSON) to original_code and return the updated code.
    """
    edits = parse_patch(patch)
    content = original_code

    for edit in edits:
        # OpenCode behavior: normalize old/new line endings to match file
        ending = _detect_line_ending(content)
        old_s = _convert_to_line_ending(_normalize_line_endings(edit.oldString), ending)
        new_s = _convert_to_line_ending(_normalize_line_endings(edit.newString), ending)

        if old_s == "":
            # OpenCode uses oldString=="" for create/overwrite.
            content = new_s
            continue

        content = replace_opencode_like(content, old_s, new_s, replaceAll=edit.replaceAll)

    return content


def demo_apply(original_code: str, patch_json: str) -> str:
    """
    Minimal helper the user requested: input original_code + patch(JSON), output applied result.
    """
    return apply_search_and_replace(original_code, patch_json)

