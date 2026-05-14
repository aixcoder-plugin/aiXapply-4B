"""
OpenCode-style prompt for a "search_and_replace" baseline.

Goal:
- Input: <source_file> (original code) + <update_snippet> (patch snippet)
- Output: JSON edits in OpenCode Edit tool style:
  [{"oldString": "...", "newString": "...", "replaceAll": false}, ...]

We intentionally keep the core constraints aligned with OpenCode's edit tool description.
"""

# NOTE: This is adapted from OpenCode's edit tool description:
# https://github.com/anomalyco/opencode/blob/6b4d617df080cef71cd8f4b041601cf47ce0edf3/packages/opencode/src/tool/edit.txt
OPENCODE_EDIT_TOOL_RULES = """Performs exact string replacements in files.

Usage:
- When editing text, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: line number + colon + space (e.g., `1: `). Everything after that space is the actual file content to match. Never include any part of the line number prefix in the oldString or newString.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `oldString` is not found in the file with an error "oldString not found in content".
- The edit will FAIL if `oldString` is found multiple times in the file with an error "Found multiple matches for oldString. Provide more surrounding lines in oldString to identify the correct match." Either provide a larger string with more surrounding context to make it unique or use `replaceAll` to change every instance of `oldString`.
- Use `replaceAll` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.
""".strip()


PROMPT_OPENCODE_SAR = {
    "system": f"""
Please produce OpenCode-style search/replace edits in json format.

OpenCode Edit tool rules (reference):
{OPENCODE_EDIT_TOOL_RULES}
""",
    "user": """<language>{language}</language>

<source_file>{source_file}</source_file>

<update_snippet>
{update_snippet}
</update_snippet>
""",
}

