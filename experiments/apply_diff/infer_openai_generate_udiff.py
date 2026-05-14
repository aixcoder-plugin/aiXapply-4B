import argparse
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from aiolimiter import AsyncLimiter
from datasets import load_dataset
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError
from tqdm import tqdm

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    # Optional dependency; environment variables can still be provided by the shell.
    pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


PROMPT_UDIFF_GENERATE = {
    "system": """You are a deterministic Unified Diff Patch Generator.

Your task:
Given a programming language, a file path, a line-numbered Source File, and a non-standard Update Snippet (with omission markers like "... existing code ..."),
produce a standard Unified Diff patch that transforms the Source File into the intended updated file.

You MUST output ONLY a Unified Diff patch (unidiff). Do NOT output the full updated file.

========================
Input Definitions
========================
- "Source File" is provided WITH line numbers.
  Each physical line has a numeric prefix and a delimiter ("123| ").
  The line number prefix is NOT part of the actual file content.
- "Update Snippet" is a partial code snippet describing changes.
  It may contain omission markers such as:
    - "... existing code ..."
    - "... rest of code ..."
    - "// ... existing code ..."
    - "# ... existing code ..."
  These markers indicate unchanged code that MUST be taken from the Source File.

========================
Required Algorithm
========================
1) Parse Source File
   - Strip the line-number prefixes to obtain the true original file content.
   - Preserve the exact original whitespace, indentation, and line endings in the content lines.

2) Expand and Interpret Update Snippet
   - Treat omission markers as placeholders for unchanged lines from the Source File.
   - Identify stable anchor lines around each omission marker and each changed region.
   - Locate the corresponding region in the Source File by exact text match on anchors.
   - The match MUST be unique. If multiple matches exist or anchors cannot be found, output nothing inside the output tags.

3) Synthesize the Updated File (internal step)
   - Apply the Update Snippet changes to the Source File to create the intended Updated File.
   - Do not leave any omission markers in the Updated File.

4) Generate the Unified Diff Patch (final output)
   - Compute a unified diff from Source File -> Updated File.
   - Emit minimal hunks that fully represent all changes.
   - Each hunk MUST include enough surrounding context to apply safely:
     - At least 3 context lines before and after each change when available.
     - Increase context if needed to make the hunk location unique.
   - Use the standard unified diff structure:
     - A two-line file header:
       --- a/<file_path>
       +++ b/<file_path>
     - One or more hunks:
       @@ -old_start,old_count +new_start,new_count @@
       <hunk lines...>

   - Hunk lines MUST:
     - Start with exactly one of: ' ' (space), '-' or '+'
     - Contain the exact code text (WITHOUT source line numbers)
     - Preserve exact indentation and whitespace.

========================
Constraints / Rules
========================
- Deterministic: Produce the same output for the same input.
- No commentary: Do not explain. Do not add analysis. Output only the patch inside the required tags.
- No laziness: Never output omission markers in the patch.
- No invented context: Context and deletions ('-') MUST exactly match the Source File content.
- No Git-only metadata: Do NOT output "index ...", "new file mode", "deleted file mode", rename/copy headers, etc.
- Single-file scope: Generate a patch ONLY for the given file.
- Safety: If you cannot confidently produce a correct patch with unique anchors, output:
  <unified_diff>
  </unified_diff>

For each hunk, the header counts MUST match the hunk body:
- Let OLD = number of hunk body lines that start with ' ' (space) or '-'.
- Let NEW = number of hunk body lines that start with ' ' (space) or '+'.
- The hunk header MUST be:
  @@ -oldStart,OLD +newStart,NEW @@

Rules:
- Always include ",count" even when count == 1 (avoid omitted counts).
- After drafting each hunk, re-count OLD/NEW and fix the header before output.
- Every hunk body line MUST start with exactly one of: ' ', '+', '-', or '\\' (for "\\ No newline at end of file").
- Context lines (' ') and removed lines ('-') MUST be copied exactly from the Source File (after removing source line-number prefixes).

========================
Output Format
========================
You MUST enclose the patch within:
<unified_diff>
...patch content...
</unified_diff>
""".strip(),
    "user": """<language>{language}</language>

<file_path>
{file_path}
</file_path>

<source_file_with_line_numbers>
{source_file_with_line_numbers}
</source_file_with_line_numbers>

<update_snippet>
{update_snippet}
</update_snippet>

Generate a Unified Diff patch that applies the update_snippet onto the source file.
Remember:
- Strip line numbers from source content; do NOT include them in the patch.
- Replace omission markers by the exact unchanged lines from the source file.
- Output ONLY the patch inside <unified_diff> tags.
""",
}


EXTENSION_MAP: Dict[str, str] = {
    "python": "py",
    "javascript": "js",
    "java": "java",
    "c++": "cpp",
    "cpp": "cpp",
    "c": "c",
    "go": "go",
    "typescript": "ts",
    "c#": "cs",
    "csharp": "cs",
    "rust": "rs",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "xml": "xml",
    "sql": "sql",
    "ini": "ini",
    "markdown": "md",
    "text": "txt",
    "html": "html",
    "restructuredtext": "rst",
    "shell": "sh",
    "bash": "sh",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
}

HUNK_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)$")
LINE_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\| ")
START_TAG = "<unified_diff>"
END_TAG = "</unified_diff>"


@dataclass(frozen=True)
class ExamplePayload:
    language: str
    file_path: str
    original_code: str
    update_snippet: str
    ground_truth: str


@dataclass(frozen=True)
class Hunk:
    source_start: int
    source_length: int
    target_start: int
    target_length: int
    lines: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a unified-diff generation model: send (line-numbered original_code + update_snippet) "
            "to an OpenAI-compatible endpoint, locally apply the predicted diff, and write jsonl records "
            "compatible with experiments/evaluation/run_evaluation.py."
        )
    )
    parser.add_argument(
        "-d",
        "--data-path",
        required=True,
        help="Path to the parquet dataset file or a datasets-compatible path.",
    )
    parser.add_argument(
        "--source-field",
        default="original_code",
        help="Dataset column containing the original full file.",
    )
    parser.add_argument(
        "--update-field",
        default="update_snippet",
        help="Dataset column containing the partial update snippet.",
    )
    parser.add_argument(
        "--target-field",
        default="final_code",
        help="Dataset column containing the ground-truth updated file.",
    )
    parser.add_argument(
        "--language-field",
        default="language",
        help="Dataset column containing the programming language.",
    )
    parser.add_argument(
        "--path-field",
        default="path",
        help="Dataset column containing the relative file path used in unified diff headers.",
    )
    parser.add_argument(
        "--filter-num",
        type=int,
        default=0,
        help="Only run the first N examples (0 means all).",
    )
    parser.add_argument(
        "-p",
        "--provider",
        default="glm-5",
        choices=["glm-5", "qwen3.5-397b", "deepseek-v3.2", "kimi-k2.5", "minimax-m2.5", "aiXapply", "custom"],
        help="Model name.",
    )
    parser.add_argument("--api-key", default=None, help="API key override (OpenAI-compatible).")
    parser.add_argument("--base-url", default=None, help="Base URL override (OpenAI-compatible).")
    parser.add_argument("--model", default=None, help="Model name override.")
    parser.add_argument(
        "--extra-body",
        default=None,
        help='Optional JSON string for request extra_body (e.g. \'{"temperature":0.3}\').',
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32 * 1024,
        help="Maximum tokens to generate for each request.",
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p nucleus sampling value.")
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=16,
        help="Maximum concurrent requests to send.",
    )
    parser.add_argument(
        "-r",
        "--rate-limit",
        type=int,
        default=120,
        help="Maximum requests allowed in the rate window.",
    )
    parser.add_argument(
        "--rate-period",
        type=int,
        default=60,
        help="Rate limit window size in seconds.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1000.0,
        help="Timeout (seconds) for each request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry attempts for recoverable errors.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output jsonl path. Default: apply_diff/predictions/{model}_generate_udiff_{timestamp}.jsonl",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Existing jsonl path to resume from. Records whose index/idx already exists "
            "will be skipped. If --output-path is omitted, resume in place."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Do not call the model; write placeholder patches and unchanged predictions.",
    )
    return parser.parse_args()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _first_non_empty(values: Iterable[Any]) -> str:
    for value in values:
        text = _as_text(value)
        if text.strip():
            return text
    return ""


def _get_from_example_or_extra(example: Dict[str, Any], *keys: str) -> str:
    extra_info = example.get("extra_info") or {}
    extra_info = extra_info if isinstance(extra_info, dict) else {}
    values: List[Any] = []
    for key in keys:
        values.append(example.get(key))
        values.append(extra_info.get(key))
    return _first_non_empty(values)


def extract_user_content(messages: Iterable[Dict[str, Any]]) -> str:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def extract_tagged_block(text: str, tags: Iterable[str]) -> str:
    for tag in tags:
        start = f"<{tag}>"
        end = f"</{tag}>"
        if start in text and end in text:
            return text.split(start, 1)[1].split(end, 1)[0]
    return ""


def build_default_file_path(language: str) -> str:
    lang_key = (language or "").lower().strip()
    ext = EXTENSION_MAP.get(lang_key, "txt")
    if ext == "dockerfile":
        return "Dockerfile"
    if ext == "makefile":
        return "Makefile"
    return f"example.{ext}"


def add_line_numbers(code: str) -> str:
    lines = code.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i + 1:>{width}}| {line}" for i, line in enumerate(lines))


def ensure_line_numbered_source(code: str) -> str:
    lines = code.splitlines()
    probe = [line for line in lines[:5] if line.strip()]
    if probe and all(LINE_NUMBER_PREFIX_RE.match(line) for line in probe):
        return code
    return add_line_numbers(code)


def build_messages(
    *,
    language: str,
    file_path: str,
    source_file: str,
    update_snippet: str,
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": PROMPT_UDIFF_GENERATE["system"]},
        {
            "role": "user",
            "content": PROMPT_UDIFF_GENERATE["user"].format(
                language=language,
                file_path=file_path,
                source_file_with_line_numbers=ensure_line_numbered_source(source_file),
                update_snippet=update_snippet,
            ),
        },
    ]


def resolve_example_payload(example: Dict[str, Any], args: argparse.Namespace) -> ExamplePayload:
    prompt = example.get("prompt") or []
    user_prompt = extract_user_content(prompt) if isinstance(prompt, list) else ""

    prompt_source = extract_tagged_block(user_prompt, ["source_file", "original_code"])
    prompt_update = extract_tagged_block(user_prompt, ["update_snippet"])
    prompt_language = extract_tagged_block(user_prompt, ["language"])
    prompt_path = extract_tagged_block(user_prompt, ["file_path", "path"])

    reward_model = example.get("reward_model") or {}
    reward_model = reward_model if isinstance(reward_model, dict) else {}

    language = _first_non_empty(
        [
            _get_from_example_or_extra(example, args.language_field),
            _get_from_example_or_extra(example, "language", "lang"),
            prompt_language,
        ]
    )
    if not language:
        language = "text"
    original_code = _first_non_empty(
        [
            _get_from_example_or_extra(example, args.source_field),
            _get_from_example_or_extra(example, "original_code", "old_code"),
            prompt_source,
        ]
    )
    update_snippet = _first_non_empty(
        [
            _get_from_example_or_extra(example, args.update_field),
            _get_from_example_or_extra(example, "update_snippet"),
            prompt_update,
        ]
    )
    ground_truth = _first_non_empty(
        [
            _get_from_example_or_extra(example, args.target_field),
            _get_from_example_or_extra(example, "final_code", "new_code", "ground_truth"),
            reward_model.get("ground_truth"),
        ]
    )
    file_path = _first_non_empty(
        [
            _get_from_example_or_extra(example, args.path_field),
            _get_from_example_or_extra(example, "path", "file_path", "relative_path_like"),
            prompt_path,
        ]
    )
    if not file_path:
        file_path = build_default_file_path(language)

    return ExamplePayload(
        language=language,
        file_path=file_path,
        original_code=original_code,
        update_snippet=update_snippet,
        ground_truth=ground_truth,
    )

sampling_one = True
async def generate_completion(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    messages: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Optional[str], float]:
    attempt = 0
    backoff_factor = 2
    global sampling_one

    while attempt <= args.max_retries:
        try:
            async with limiter:
                start_time = time.time()
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=args.model,
                        stream=False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        messages=messages,
                        extra_body=args.extra_body or {}
                    ),
                    timeout=args.request_timeout,
                )
                end_time = time.time()
            if sampling_one:
                logger.info(f"Sampling one response: \n{response}")
                sampling_one = False
            return response.choices[0].message.content, end_time - start_time
        except asyncio.TimeoutError:
            logger.error("Request timed out after %.1f seconds", args.request_timeout)
        except APITimeoutError:
            logger.error("OpenAI API request timed out")
        except APIConnectionError as ce:
            logger.error("Connection error (server disconnected / network issue): %s", ce)
        except APIStatusError as se:
            status = se.status_code
            logger.error("HTTP error %s: %s", status, se.message)
            if status not in {429, 500, 502, 503}:
                return None, 0.0
        except Exception:
            logger.exception("Unexpected error while requesting completion")

        attempt += 1
        if attempt <= args.max_retries:
            wait_time = backoff_factor**attempt + 10
            logger.info(
                "Retrying in %s seconds (attempt %s/%s)",
                wait_time,
                attempt,
                args.max_retries,
            )
            await asyncio.sleep(wait_time)
        else:
            return None, 0.0
    return None, 0.0


def strip_outer_code_fence(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    first_idx = 0
    while first_idx < len(lines) and lines[first_idx].strip() == "":
        first_idx += 1
    if first_idx >= len(lines):
        return text
    if not lines[first_idx].lstrip().startswith("```"):
        return text

    last_idx = len(lines) - 1
    while last_idx >= 0 and lines[last_idx].strip() == "":
        last_idx -= 1
    if last_idx <= first_idx:
        return text
    if lines[last_idx].strip() != "```":
        return text

    return "\n".join(lines[first_idx + 1 : last_idx])


def looks_like_unified_diff(text: str) -> bool:
    has_hunk = re.search(r"(?m)^@@\s+-\d+", text) is not None
    has_headers = (
        re.search(r"(?m)^---\s+", text) is not None and re.search(r"(?m)^\+\+\+\s+", text) is not None
    )
    return has_hunk or has_headers


def extract_unified_diff(response: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(response, str) or not response.strip():
        return None, "Empty model output"

    normalized = response.replace("\r\n", "\n").replace("\r", "\n")
    start_idx = normalized.find(START_TAG)
    end_idx = normalized.rfind(END_TAG)
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx + len(START_TAG):
        patch_text = normalized[start_idx + len(START_TAG) : end_idx]
    elif looks_like_unified_diff(normalized):
        patch_text = normalized
    else:
        return None, "No <unified_diff> block or unified diff content found"

    patch_text = strip_outer_code_fence(patch_text).strip("\n")
    if not patch_text:
        return None, "No patch content found"
    if not patch_text.endswith("\n"):
        patch_text += "\n"
    return patch_text, None


def parse_unified_diff(patch_text: str) -> Tuple[List[Hunk], Optional[str]]:
    normalized = patch_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.splitlines()

    if len(re.findall(r"(?m)^diff --git\s+", normalized)) > 1:
        return [], "Expected single-file patch, got multiple diff --git sections"

    hunks: List[Hunk] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = HUNK_HEADER_RE.match(line)
        if not match:
            i += 1
            continue

        source_start = int(match.group(1))
        source_length = int(match.group(2) or "1")
        target_start = int(match.group(3))
        target_length = int(match.group(4) or "1")

        i += 1
        body: List[str] = []
        while i < len(lines):
            next_line = lines[i]
            if HUNK_HEADER_RE.match(next_line):
                break
            if next_line.startswith((" ", "+", "-", "\\")):
                body.append(next_line)
                i += 1
                continue
            if next_line.startswith(("diff --git ", "--- ", "+++ ")):
                break
            break

        if not body:
            return [], f"Empty hunk body near source line {source_start}"

        hunks.append(
            Hunk(
                source_start=source_start,
                source_length=source_length,
                target_start=target_start,
                target_length=target_length,
                lines=body,
            )
        )

    return hunks, None


def find_unique_sublist(haystack: List[str], needle: List[str], hint: Optional[int] = None) -> Optional[int]:
    if not needle:
        if hint is None:
            return 0
        return max(0, min(hint, len(haystack)))

    n = len(needle)

    def matches_at(pos: Optional[int]) -> bool:
        if pos is None:
            return False
        if pos < 0 or pos + n > len(haystack):
            return False
        return haystack[pos : pos + n] == needle

    if hint is not None and matches_at(hint):
        return hint

    first = needle[0]
    candidates: List[int] = []
    max_i = len(haystack) - n
    for idx in range(max_i + 1):
        if haystack[idx] == first and haystack[idx : idx + n] == needle:
            candidates.append(idx)

    if len(candidates) == 1:
        return candidates[0]
    return None


def apply_unified_diff(original_code: str, patch_text: str) -> Tuple[Optional[str], Optional[str]]:
    hunks, parse_error = parse_unified_diff(patch_text)
    if parse_error:
        return None, parse_error
    if not hunks:
        return original_code.replace("\r\n", "\n").replace("\r", "\n"), None

    lines = original_code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    offset = 0

    for hunk in hunks:
        old_block: List[str] = []
        new_block: List[str] = []

        for raw_line in hunk.lines:
            prefix = raw_line[:1]
            if prefix == "\\":
                continue

            text = raw_line[1:]
            if prefix in {" ", "-"}:
                old_block.append(text)
            if prefix in {" ", "+"}:
                new_block.append(text)

        hint = (hunk.source_start - 1) + offset
        pos = find_unique_sublist(lines, old_block, hint=hint)
        if pos is None:
            return None, (
                "Hunk failed to apply (no unique match). "
                f"Hint line={hint + 1}, "
                f"hunk_header=-{hunk.source_start},{hunk.source_length} "
                f"+{hunk.target_start},{hunk.target_length}"
            )

        lines[pos : pos + len(old_block)] = new_block
        offset += len(new_block) - len(old_block)

    return "\n".join(lines), None


def wrap_update_file(content: Optional[str]) -> str:
    if content is None:
        return "<update_file></update_file>"
    return f"<update_file>{content}</update_file>"


def build_record(
    *,
    idx: int,
    example: Dict[str, Any],
    payload: ExamplePayload,
    prediction_patch: Optional[str],
    applied_code: Optional[str],
    apply_error: Optional[str],
    elapsed_time: float,
) -> Dict[str, Any]:
    return {
        "index": int(idx),
        "prediction": wrap_update_file(applied_code),
        "elapsed_time": float(elapsed_time),
        "ground_truth": payload.ground_truth,
        "original_code": payload.original_code,
        "update_snippet": payload.update_snippet,
        "language": payload.language,
        "prediction_patch": prediction_patch if isinstance(prediction_patch, str) else "",
        "apply_error": apply_error,
        "path": payload.file_path,
        "repo": example.get("repo"),
        "commit": example.get("commit"),
        "message": example.get("message"),
        "license": example.get("license"),
        "change_kind": example.get("change_kind"),
        "n_added": example.get("n_added"),
        "n_removed": example.get("n_removed"),
        "n_hunks": example.get("n_hunks"),
    }


async def process_example(
    idx: int,
    example: Dict[str, Any],
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Any]]:
    payload = resolve_example_payload(example, args)

    if args.dry_run:
        record = build_record(
            idx=idx,
            example=example,
            payload=payload,
            prediction_patch="<unified_diff></unified_diff>",
            applied_code=payload.original_code,
            apply_error=None,
            elapsed_time=0.0,
        )
        return idx, record

    messages = build_messages(
        language=payload.language,
        file_path=payload.file_path,
        source_file=payload.original_code,
        update_snippet=payload.update_snippet,
    )
    prediction_patch, elapsed_time = await generate_completion(client, limiter, messages, args)

    applied_code: Optional[str] = None
    apply_error: Optional[str] = None
    patch_text: Optional[str] = None

    if prediction_patch:
        patch_text, apply_error = extract_unified_diff(prediction_patch)
        if patch_text is not None:
            applied_code, apply_error = apply_unified_diff(payload.original_code, patch_text)
    else:
        apply_error = "Empty model output"

    record = build_record(
        idx=idx,
        example=example,
        payload=payload,
        prediction_patch=prediction_patch,
        applied_code=applied_code,
        apply_error=apply_error,
        elapsed_time=elapsed_time,
    )
    return idx, record


def _load_dataset_auto(data_path: Path):
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")
    if data_path.is_file() and data_path.suffix.lower() == ".parquet":
        return load_dataset("parquet", data_files={"data": str(data_path)})["data"]
    return load_dataset(str(data_path))["train"]


def _parse_extra_body(extra_body: Optional[str]) -> Dict[str, Any]:
    if not extra_body:
        return {"enable_thinking": False, "thinking": {"type": "disabled"}}
    try:
        obj = json.loads(extra_body)
    except Exception as e:
        raise ValueError(f"--extra-body must be valid JSON. got error: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("--extra-body must be a JSON object.")
    return {"enable_thinking": False, "thinking": {"type": "disabled"}, **obj}


def _extract_record_index(record: Dict[str, Any]) -> Optional[int]:
    raw_index = record.get("index")
    if raw_index is None:
        raw_index = record.get("idx")
    if raw_index is None:
        return None
    try:
        return int(raw_index)
    except (TypeError, ValueError):
        return None


def _record_has_prediction_output(record: Dict[str, Any]) -> bool:
    prediction_patch = record.get("prediction_patch")
    return isinstance(prediction_patch, str) and bool(prediction_patch.strip())


def _load_resume_state(resume_path: Path, dataset_size: int) -> Tuple[set[int], List[str]]:
    if not resume_path.exists():
        raise FileNotFoundError(f"Could not find resume file at {resume_path}")

    completed_indices: set[int] = set()
    preserved_lines: List[str] = []
    invalid_count = 0
    duplicate_count = 0
    out_of_range_count = 0
    retryable_count = 0

    with resume_path.open("r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_count += 1
                continue
            if not isinstance(record, dict):
                invalid_count += 1
                continue

            idx = _extract_record_index(record)
            if idx is None:
                invalid_count += 1
                continue
            if idx < 0 or idx >= dataset_size:
                out_of_range_count += 1
                continue
            if not _record_has_prediction_output(record):
                retryable_count += 1
                continue
            if idx in completed_indices:
                duplicate_count += 1
                continue

            completed_indices.add(idx)
            preserved_lines.append(line)

    logger.info(
        "Loaded %s completed records from %s",
        len(completed_indices),
        resume_path,
    )
    if duplicate_count:
        logger.warning("Ignored %s duplicate indices while loading %s", duplicate_count, resume_path)
    if invalid_count:
        logger.warning("Ignored %s invalid resume lines while loading %s", invalid_count, resume_path)
    if out_of_range_count:
        logger.warning("Ignored %s out-of-range indices while loading %s", out_of_range_count, resume_path)
    if retryable_count:
        logger.info(
            "Will retry %s existing records with empty prediction output from %s",
            retryable_count,
            resume_path,
        )

    return completed_indices, preserved_lines


def _path_needs_leading_newline(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    with path.open("rb") as infile:
        infile.seek(-1, os.SEEK_END)
        return infile.read(1) != b"\n"


async def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    dataset = _load_dataset_auto(data_path)

    if args.filter_num > 0:
        dataset = dataset.select(range(min(args.filter_num, len(dataset))))
        logger.info("Filtered dataset to %s examples", len(dataset))

    resume_path = Path(args.resume_from) if args.resume_from else None
    if args.output_path:
        output_path = Path(args.output_path)
    elif resume_path is not None:
        output_path = resume_path
    else:
        output_path = SCRIPT_DIR / "predictions" / f"{args.model}_generate_udiff_{run_timestamp}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("output_path:", str(output_path), flush=True)

    completed_indices: set[int] = set()
    preserved_lines: List[str] = []
    output_mode = "w"
    needs_leading_newline = False

    if resume_path is not None:
        completed_indices, preserved_lines = _load_resume_state(resume_path, len(dataset))
        logger.info(
            "Resume mode: %s/%s examples already completed, %s remaining",
            len(completed_indices),
            len(dataset),
            len(dataset) - len(completed_indices),
        )

        with output_path.open("w", encoding="utf-8") as out_file:
            for line in preserved_lines:
                out_file.write(line + "\n")
        output_mode = "a"
        needs_leading_newline = _path_needs_leading_newline(output_path)

        if len(completed_indices) >= len(dataset):
            logger.info("All %s examples are already present in %s", len(dataset), resume_path)
            return

    limiter = AsyncLimiter(args.rate_limit, args.rate_period)
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    progress = tqdm(total=len(dataset), desc="Requesting(GEN_UDIFF)", initial=len(completed_indices))
    buffer: Dict[int, Dict[str, Any]] = {}
    next_to_write = 0
    pending: set[asyncio.Task] = set()
    task_to_index: Dict[asyncio.Task, int] = {}
    index_iter = iter(range(len(dataset)))

    def advance_next_to_write() -> None:
        nonlocal next_to_write
        while next_to_write in completed_indices:
            next_to_write += 1

    def enqueue() -> bool:
        while True:
            try:
                index = next(index_iter)
            except StopIteration:
                return False
            if index not in completed_indices:
                break
        task = asyncio.create_task(process_example(index, dataset[index], client, limiter, args))
        pending.add(task)
        task_to_index[task] = int(index)
        return True

    for _ in range(min(args.concurrency, len(dataset))):
        enqueue()

    with output_path.open(output_mode, encoding="utf-8") as out_file:
        if needs_leading_newline:
            out_file.write("\n")
            out_file.flush()
        advance_next_to_write()
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                idx_hint = task_to_index.pop(task, None)
                try:
                    idx, record = task.result()
                except Exception as e:
                    logger.exception("Task failed at idx=%s", idx_hint)
                    if idx_hint is None:
                        progress.update(1)
                        continue
                    try:
                        example = dataset[int(idx_hint)]
                    except Exception:
                        example = {}
                    payload = resolve_example_payload(example, args)
                    record = build_record(
                        idx=int(idx_hint),
                        example=example,
                        payload=payload,
                        prediction_patch="",
                        applied_code=None,
                        apply_error=f"Inference error: {e}",
                        elapsed_time=0.0,
                    )
                    record["inference_error"] = repr(e)
                    idx = int(idx_hint)

                buffer[idx] = record
                while next_to_write in buffer:
                    current = buffer.pop(next_to_write)
                    out_file.write(json.dumps(current, ensure_ascii=False) + "\n")
                    out_file.flush()
                    next_to_write += 1
                    advance_next_to_write()
                progress.update(1)

            while len(pending) < args.concurrency and enqueue():
                continue

    progress.close()
    logger.info("Generated-diff predictions written to %s", output_path)


def main() -> None:
    args = parse_args()

    model_configs = {
         "glm-5": {
            "api_key": os.getenv("OPENAI_KEY", ""),
            "base_url": "https://api.z.ai/api/paas/v4/",
            "model": "glm-5",
        },
        "qwen3.5-397b": {
            "api_key": os.getenv("OPENAI_KEY", ""),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3.5-397b-a17b",
        },
        "deepseek-v3.2": {
            "api_key": os.getenv("OPENAI_KEY", ""),
            "base_url": os.getenv("VOLCANO_ENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            "model": "deepseek-v3-2-251201",
        },
        "kimi-k2.5": {
            "api_key": os.getenv("OPENAI_KEY", ""),
            "base_url": "https://api.moonshot.cn/v1",
            "model": "kimi-k2.5",
        },
        "minimax-m2.5": {
            "api_key": os.getenv("OPENAI_KEY", ""),
            "base_url": "https://api.minimaxi.com/v1",
            "model": "MiniMax-M2.5",
        },
        "aiXapply": {
            "api_key": os.getenv("OPENAI_KEY", "EMPTY"),
            "base_url": os.getenv("GENERATE_UDIFF_LOCAL_BASE_URL", "http://127.0.0.1:14123/v1"),
            "model": "aiXapply",
        },
        "custom": {},
    }

    preset = model_configs.get(args.provider, {})

    args.api_key = args.api_key if args.api_key is not None else preset.get("api_key", "")
    args.base_url = args.base_url if args.base_url is not None else preset.get("base_url", "")
    args.model = args.model if args.model is not None else preset.get("model", "")
    args.extra_body = _parse_extra_body(args.extra_body) or preset.get("extra_body", {})

    if args.model.startswith("kimi"):
        args.temperature = 0.3
        args.top_p = 0.8

    if not args.api_key:
        raise ValueError("api_key is empty. Provide --api-key or choose a provider preset.")
    if not args.base_url:
        raise ValueError("base_url is empty. Provide --base-url or choose a provider preset.")
    if not args.model:
        raise ValueError("model is empty. Provide --model or choose a provider preset.")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
