from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import re
import json
from pygments import lex
from pygments.lexers import guess_lexer, get_lexer_by_name
from pygments.token import Token

import yaml

from xml.etree import ElementTree as ET
from configparser import ConfigParser

MAX_WORKERS = 16

# ============== Error Type Penalty Configuration ==============
PATCH_ERROR_PENALTY = 0.25      # Penalty per patch error
SIDE_EFFECT_PENALTY = 0.35      # Penalty per side-effect error (heavier)
MIN_REWARD = 0                # Minimum score

# ============== Language Name to Pygments Lexer Mapping ==============
# Custom language name -> Pygments lexer name
LANGUAGE_TO_LEXER = {
    # Programming languages
    "python": "python",
    "javascript": "javascript",
    "java": "java",
    "c++": "cpp",
    "cpp": "cpp",
    "c": "c",
    "go": "go",
    "typescript": "typescript",
    "c#": "csharp",
    "csharp": "csharp",
    "rust": "rust",
    
    # Data/configuration formats
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "xml": "xml",
    "sql": "sql",
    "ini": "ini",
    
    # Markup/documentation languages
    "markdown": "markdown",
    "md": "markdown",
    "text": "text",
    "txt": "text",
    "html": "html",
    "restructuredtext": "rst",
    "rst": "rst",
    
    # Scripting/build
    "shell": "bash",
    "bash": "bash",
    "sh": "bash",
    "dockerfile": "docker",
    "docker": "docker",
    "makefile": "make",
    "make": "make",
}


def normalize_language(lang: str) -> str:
    """Normalize language name to Pygments lexer name"""
    if lang is None:
        return None
    lang_lower = lang.lower().strip()
    return LANGUAGE_TO_LEXER.get(lang_lower, lang_lower)


def strip_update_tags(s: str) -> str:
    start_tag = "<update_file>"
    end_tag = "</update_file>"
    
    start_idx = s.find(start_tag)
    if start_idx != -1:
        s = s[start_idx + len(start_tag):]
    
    end_idx = s.rfind(end_tag)
    if end_idx != -1:
        s = s[:end_idx]
    
    return s


# ============== Non-programming Language Comparison Functions ==============

def is_same_json(code_a: str, code_b: str) -> bool:
    """
    JSON parsing-based comparison.
    Ignores whitespace and formatting differences, only compares data structures.
    """
    try:
        obj_a = json.loads(code_a)
        obj_b = json.loads(code_b)
        return obj_a == obj_b
    except json.JSONDecodeError:
        # If parsing fails, fall back to string comparison
        return code_a.strip() == code_b.strip()


def is_same_yaml(code_a: str, code_b: str) -> bool:
    """
    YAML parsing-based comparison.
    Ignores formatting differences, only compares data structures.
    """

    try:
        # Use safe_load_all to handle multi-document YAML
        docs_a = list(yaml.safe_load_all(code_a))
        docs_b = list(yaml.safe_load_all(code_b))
        return docs_a == docs_b
    except yaml.YAMLError:
        return code_a.strip() == code_b.strip()


def _element_to_tuple(elem):
    """Convert XML element to a comparable tuple structure (ignoring whitespace-only text)"""
    children = tuple(_element_to_tuple(child) for child in elem)
    attribs = tuple(sorted(elem.attrib.items()))
    
    # Handle text: treat whitespace-only as None
    text = elem.text.strip() if elem.text and elem.text.strip() else None
    # tail is usually text after the element; for structural comparison,
    # whitespace-only tail is typically ignored, but non-whitespace content is preserved
    tail = elem.tail.strip() if elem.tail and elem.tail.strip() else None
    
    return (elem.tag, attribs, text, children)


def is_same_xml(code_a: str, code_b: str) -> bool:
    """
    XML DOM parsing-based comparison.
    Ignores whitespace differences, compares XML structure.
    """

    try:
        tree_a = ET.fromstring(code_a.strip())
        tree_b = ET.fromstring(code_b.strip())
        
        # Convert to comparable structures
        tuple_a = _element_to_tuple(tree_a)
        tuple_b = _element_to_tuple(tree_b)
        
        return tuple_a == tuple_b
    except ET.ParseError:
        return code_a.strip() == code_b.strip()


def is_same_html(code_a: str, code_b: str) -> bool:
    """
    HTML parsing-based comparison.
    HTML is more lenient than XML; simplified processing is used here.
    """
    # Function to normalize HTML
    def normalize_html(html):
        # Remove excessive whitespace
        html = re.sub(r'\s+', ' ', html)
        # Remove whitespace between tags
        html = re.sub(r'>\s+<', '><', html)
        # Remove excessive whitespace around self-closing tags
        html = re.sub(r'\s*/\s*>', '/>', html)
        return html.strip().lower()
    
    # First try comparing after normalization
    if normalize_html(code_a) == normalize_html(code_b):
        return True
    
    # Try parsing as XML (for strict XHTML)
    try:
        return is_same_xml(code_a, code_b)
    except Exception:
        return False


def is_same_ini(code_a: str, code_b: str) -> bool:
    """
    INI configuration parsing-based comparison.
    Ignores whitespace, comment differences, and section order.
    """

    try:
        parser_a = ConfigParser()
        parser_b = ConfigParser()
        
        parser_a.read_string(code_a)
        parser_b.read_string(code_b)
        
        # Compare all sections (ignoring order)
        if set(parser_a.sections()) != set(parser_b.sections()):
            return False
        
        for section in parser_a.sections():
            if dict(parser_a.items(section)) != dict(parser_b.items(section)):
                return False
        
        # Compare DEFAULT section
        if dict(parser_a.defaults()) != dict(parser_b.defaults()):
            return False
        
        return True
    except Exception:
        return code_a.strip() == code_b.strip()


def is_same_plain_text(code_a: str, code_b: str) -> bool:
    """
    Plain text comparison.
    Only performs basic whitespace normalization.
    """
    # Normalize line endings and trailing whitespace
    def normalize_text(text):
        lines = text.splitlines()
        # Strip trailing whitespace from each line, but preserve line structure
        lines = [line.rstrip() for line in lines]
        # Remove trailing empty lines
        while lines and not lines[-1]:
            lines.pop()
        return '\n'.join(lines)
    
    return normalize_text(code_a) == normalize_text(code_b)


def is_same_markdown(code_a: str, code_b: str) -> bool:
    """
    Markdown comparison.
    Ignores blank line differences, but preserves content structure.
    """
    def normalize_markdown(md):
        lines = md.splitlines()
        # Strip trailing whitespace from each line
        lines = [line.rstrip() for line in lines]
        # Merge multiple consecutive blank lines into one
        result = []
        prev_empty = False
        for line in lines:
            is_empty = not line.strip()
            if is_empty:
                if not prev_empty:
                    result.append('')
                prev_empty = True
            else:
                result.append(line)
                prev_empty = False
        # Remove leading and trailing blank lines
        while result and not result[0]:
            result.pop(0)
        while result and not result[-1]:
            result.pop()
        return '\n'.join(result)
    
    return normalize_markdown(code_a) == normalize_markdown(code_b)


def is_same_sql(code_a: str, code_b: str) -> bool:
    """
    SQL comparison.
    Ignores whitespace and case differences (for keywords), but preserves comment content for comparison.
    """
    def normalize_sql(sql):
        # Normalize whitespace (but preserve comment content)
        sql = re.sub(r'\s+', ' ', sql)
        return sql.strip().lower()
    
    return normalize_sql(code_a) == normalize_sql(code_b)


def get_tokens(code: str, lang: str = None) -> list:
    """
    Get the token list of the code.
    
    Args:
        code: Source code string
        lang: Language name (will be automatically normalized)
    
    Returns:
        Token list, or None if parsing fails
    """
    # Normalize language name
    normalized_lang = normalize_language(lang)
    
    try:
        if normalized_lang:
            lexer = get_lexer_by_name(normalized_lang)
        else:
            lexer = guess_lexer(code)
    except Exception:
        return None
    
    tokens = []
    for token_type, token_value in lex(code, lexer):
        # To skip comments: token_type in Token.Comment
        if token_type in Token.Text.Whitespace:
            continue
        # Skip pure whitespace characters
        if not token_value.strip():
            continue
        tokens.append((token_type, token_value))
    return tokens


def is_same_code_pygments(code_a: str, code_b: str, lang: str = None) -> bool:
    """Pygments token-based code comparison, supports multiple languages"""
    tokens_a = get_tokens(code_a, lang)
    tokens_b = get_tokens(code_b, lang)
    if tokens_a is None or tokens_b is None:
        return code_a.strip() == code_b.strip()
    return tokens_a == tokens_b

# Normalize line endings during data preprocessing
def normalize_line_endings(text):
    return text.replace('\r\n', '\n').replace('\r', '\n')

def is_same_code(code_a: str, code_b: str, lang: str = None) -> bool:
    """
    Universal code comparison function that selects the best comparison strategy based on language.
    
    Supported language types and comparison strategies:
    - JSON/YAML: Parsed data structure comparison
    - XML/HTML: DOM structure comparison
    - INI: Configuration parsing comparison
    - Markdown/Text/RST: Normalized text comparison
    - SQL: Normalized comparison (ignoring whitespace and case)
    - Other programming languages: Pygments token comparison
    
    Args:
        code_a: First code segment
        code_b: Second code segment
        lang: Language name (optional)
    
    Returns:
        True if the two code segments are semantically identical
    """
    normalized_lang = normalize_language(lang)

    # Convert Windows-style \r\n to Unix-style \n; mainly affects ini files
    code_a = normalize_line_endings(code_a)
    code_b = normalize_line_endings(code_b)
    
    # JSON: use parsed comparison
    if normalized_lang == "json":
        return is_same_json(code_a, code_b)
    
    # YAML: use parsed comparison
    if normalized_lang in ("yaml", "yml"):
        return is_same_yaml(code_a, code_b)
    
    # XML: use DOM comparison
    if normalized_lang == "xml":
        return is_same_xml(code_a, code_b)
    
    # SQL: use normalized comparison
    if normalized_lang == "sql":
        return is_same_sql(code_a, code_b)

    # INI: use configuration parsing comparison
    if normalized_lang == "ini":
        return is_same_ini(code_a, code_b)
    
    # Markdown: use special processing
    if normalized_lang in ("markdown", "md"):
        return is_same_markdown(code_a, code_b)
    
    # Plain text and RST: use text comparison
    if normalized_lang in ("text", "txt", "rst", "restructuredtext"):
        return is_same_plain_text(code_a, code_b)
    
    # HTML: use lenient DOM comparison
    if normalized_lang == "html":
        return is_same_html(code_a, code_b)

    # Other languages (programming languages, scripts, etc.): use Pygments token comparison
    return is_same_code_pygments(code_a, code_b, lang)


def analyze_error_types(update_snippet: str, reference: str, prediction: str):
    """
    Analyze the types of prediction errors.
    
    Returns: (patch_error_count, side_effect_count)
    - patch_error_count: Number of patch anchor/content errors (errors within the snippet region)
    - side_effect_count: Number of side effects outside the patch region (modifications to areas that should not be changed)
    """
    def to_lines(text):
        return [line.strip() for line in text.strip().splitlines()]
    
    ref_lines = to_lines(reference)
    pred_lines = to_lines(prediction)
    snippet_lines = set(to_lines(update_snippet)) if update_snippet else set()
    
    matcher = SequenceMatcher(None, ref_lines, pred_lines)
    
    patch_error_count = 0
    side_effect_count = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        
        ref_segment = ref_lines[i1:i2]
        pred_segment = pred_lines[j1:j2]
        
        # Classify each line as either a patch error or a side effect
        if tag in ('replace', 'delete'):
            for line in ref_segment:
                if line and line in snippet_lines:
                    patch_error_count += 1
                elif line:
                    side_effect_count += 1
        
        if tag == 'insert':
            for line in pred_segment:
                if line and line in snippet_lines:
                    patch_error_count += 1
                elif line:
                    side_effect_count += 1
    
    return patch_error_count, side_effect_count


def compute_score_pygments(predictions, reference_answer, lang=None):
    predictions = strip_update_tags(predictions)
    
    if is_same_code(predictions, reference_answer, lang):
        reward_score = 1.0
    else:
        reward_score = 0.0

    return reward_score


def compute_score_with_error_types(prediction: str, reference: str, update_snippet: str = None, lang: str = None, split: str = None):
    """
    Scoring function based on error types.
    
    - Exact match: 1.0
    - When errors exist, deduct based on error type:
      - Patch error: deduct 0.25 each
      - Side-effect error: deduct 0.35 each (more severe)
    """
    prediction = strip_update_tags(prediction)
    
    # Exact match check
    if is_same_code(prediction, reference, lang):
        return 1.0
    elif split == "test":
        return 0
    
    # Analyze error types
    patch_errors, side_effects = analyze_error_types(update_snippet, reference, prediction)

    # Calculate score
    score = 1.0 - (patch_errors * PATCH_ERROR_PENALTY) - (side_effects * SIDE_EFFECT_PENALTY)
    
    return max(MIN_REWARD, score)


def _compute_score_batch_impl(
    score_fn,
    data_sources=None,
    solution_strs=None,
    ground_truths=None,
    extra_infos=None,
    **kwargs,
):
    """
    Generic batch scoring function.
    
    Args:
        score_fn: Scoring function with signature (solution_str, ground_truth, extra_info) -> score
    """
    if solution_strs is None:
        solution_strs = kwargs.get("solution_str")
        ground_truths = kwargs.get("ground_truth")
        extra_infos = kwargs.get("extra_info")

        if not isinstance(solution_strs, list):
            solution_strs = [solution_strs]
        if not isinstance(ground_truths, list):
            ground_truths = [ground_truths]
        if extra_infos is not None and not isinstance(extra_infos, list):
            extra_infos = [extra_infos]
        return_single = True
    else:
        return_single = False

    # If no extra_infos, create a list of empty dicts
    if extra_infos is None:
        extra_infos = [{} for _ in solution_strs]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(score_fn, sol, gt, info)
            for sol, gt, info in zip(solution_strs, ground_truths, extra_infos, strict=True)
        ]
        results = [f.result() for f in futures]

    return results[0] if return_single else results
 

def _score_pygments(solution_str, ground_truth, extra_info):
    """Pygments scoring: token-based comparison, supports multiple languages"""
    lang = extra_info.get("language") if extra_info else None
    return compute_score_pygments(solution_str, ground_truth, lang)


def _score_dense_reward(solution_str, ground_truth, extra_info):
    """Dense reward scoring: distinguishes between patch errors and side-effect errors"""
    lang = extra_info.get("language") if extra_info else None
    snippet = extra_info.get("update_snippet") if extra_info else None
    split = extra_info.get("split") if extra_info else None
    return compute_score_with_error_types(solution_str, ground_truth, snippet, lang, split)


def compute_score_batch_pygments(data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None, **kwargs):
    """
    Batch scoring - Pygments version
    """
    return _compute_score_batch_impl(_score_pygments, data_sources, solution_strs, ground_truths, extra_infos, **kwargs)


def compute_score_batch_dense_reward(data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None, **kwargs):
    """
    Batch scoring - Dense reward version
    
    extra_info may contain:
    - language: Programming language
    - update_snippet: Update snippet (used to distinguish patch errors from side-effect errors)
    """
    return _compute_score_batch_impl(_score_dense_reward, data_sources, solution_strs, ground_truths, extra_infos, **kwargs)
