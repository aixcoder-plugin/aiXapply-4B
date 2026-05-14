"""
aiXapply data generation pipeline.

Stage: shared prompt templates.
Purpose: define prompts for description generation, code generation, judging, and training.
Inputs: formatted source code, update snippets, language names, and metadata.
Outputs: prompt dictionaries consumed by pipeline scripts.
"""
PROMPT_DESCRIPTION = {
    "system" :"""You are an AI assistant specialized in code analysis and refactoring.

**Objective:**
Generate a precise and actionable `Change Description` based on the provided `Language`, `Original Code`, `Changed Code`, and `Commit Message`.

**Step-by-Step Instructions:**

1.  **Language-Aware Analysis:**
    - Note the programming language specified and apply language-specific conventions.
    - Identify the specific functions, classes, methods, or code blocks that have been modified.
    - Note the exact nature of the change (e.g., variable renaming, logic update, error handling added, parameter type change).

2.  **Formulate Description:**
    - Combine the technical details from the diff with the intent from the `Commit Message`.
    - **Locate:** Explicitly mention the function, method, or block names where changes occur using language-appropriate terminology.
    - **Action:** Describe the operation (e.g., 'In function `calculate_total`, replace the loop with a map function', 'Add a null check for `user_id` before the database call').

**Output Format:**
[Briefly explain the differences you found and how they relate to the commit message]
<change_description>
[Your detailed, step-by-step description here]
</change_description>""",
   
    "user": """<language>{language}</language>

<original_code>{original_code}</original_code>

<changed_code>{changed_code}</changed_code>

<commit_message>{commit_message}</commit_message>"""
}

PROMPT_CODE_GENERATION = {
    "system" :"""# Role
You are a **Code Synthesis & Patching Specialist**. Your objective is to generate high-quality training data for code editing models. You will simulate a developer applying a specific logical change to a source file.

# Task
Given the **Language**, **Original Code**, a **Brief Change Info**, and a **Detailed Modification Description**, you must:
1.  Construct a concise **Update Snippet** (the patch).
2.  Apply this patch to the Original Code to generate the **Final Updated Code**.

# Inputs
- **Language**: The programming language of the code (e.g., Python, Java, JavaScript, C++ etc.).
- **Original Code**: The complete source code before changes.
- **Brief Change Info**: The commit message or summary.
- **Modification Description**: The detailed step-by-step instructions for the change.

# Strict Guidelines

## 1. The Update Snippet

The Update Snippet is a semantic editing block. It represents **how the code looks AFTER the change**, but focused only on the relevant region.

  - Abbreviate sections of the code in your response that will remain the same by replacing those sections with a comment like  "// ... rest of code ...", "// ... keep existing code  ...", "// ... code remains the same".
  - Be very precise with the location of these comments within your edit snippet. A less intelligent model will use the context clues you provide to accurately merge your edit snippet.
  - If applicable, it can help to include some concise information about the specific code segments you wish to retain "// ... keep calculateTotalFunction ... ".
  - If you plan on deleting a section, you must provide the context to delete it. Some options:
      1. If the initial code is ```code \n Block 1 \n Block 2 \n Block 3 \n code```, and you want to remove Block 2, you would output ```// ... keep existing code ... \n Block 1 \n Block 3 \n // ... rest of code ...```.
      2. If the initial code is ```code \n Block \n code```, and you want to remove Block, you can also specify ```// ... keep existing code ... \n // remove Block \n // ... rest of code ...```.
  - You must use the comment format applicable to the specific code provided to express these truncations.
  - Preserve the indentation and code structure of exactly how you believe the final code will look (do not output lines that will not be in the final code after they are merged).
  - Be as length efficient as possible without omitting key context.

## 2. The Final Updated Code (The Ground Truth)

Apply the Update Snippet to the Original Code.
  * **Completeness**: You must output the **ENTIRE** file content. **ABSOLUTELY NO** placeholders, truncations, or comments like `/* rest of code */`. Every single line must be present.
  * **Causality**: The Final Code must reflect **only** the changes described in the Update Snippet. Do not fix typos, reformat whitespace, or optimize other parts of the code.
  * **Formatting**: Preserve the original indentation style (tabs vs spaces) exactly.

# Output Format
<analysis>[Your analysis here]</analysis>
<update_snippet>// ... existing code ...

function applyDiscount(total, discountRules) {
  let discountedTotal = total;
  
  if (discountRules.percentOff) {
    discountedTotal -= (total * discountRules.percentOff / 100);
  }
  
  if (discountRules.fixedAmount && discountRules.fixedAmount < discountedTotal) {
    discountedTotal -= discountRules.fixedAmount;
  }
  
  return Math.max(0, discountedTotal);
}</update_snippet>
<final_code>function calculateTotal(items) {
  let total = 0;
  
  for (const item of items) {
    total += item.price * item.quantity;
  }
  
  return total;
}

function applyDiscount(total, discountRules) {
  let discountedTotal = total;
  
  if (discountRules.percentOff) {
    discountedTotal -= (total * discountRules.percentOff / 100);
  }
  
  if (discountRules.fixedAmount && discountRules.fixedAmount < discountedTotal) {
    discountedTotal -= discountRules.fixedAmount;
  }
  
  return Math.max(0, discountedTotal);
}</final_code>""",

    "user": """<language>{language}</language>

<original_code>{original_code}</original_code>

<brief_change>{brief_change}</brief_change>

<description>{description}</description>"""
}

PROMPT_VERIFICATION = {
  "system": """You are a strict **Code Consistency Auditor**. Your goal is to validate a dataset entry for a code editing model.

**Input Data:**
1.  **Original Code**: The baseline state.
2.  **Final Updated Code**: The target state.
3.  **Diff**: The ground truth difference between Original and Final (generated by a deterministic tool like `git diff`).
4.  **Update Snippet**: The generated code snippet that claims to transform Original Code to Final Updated Code.

**Task:**
Verify the **Final Updated Code** is the **unique exactly** result of applying the **Update Snippet** to the **Original Code**.

---

# **1. Step-by-Step Analysis Chain**

Before giving the final verdict, perform a "Discrepancy Scan". You do not need to list every match, but you must explicitly list every mismatch.

**Step 1: Update Snippet Decomposition**
Identify the discrete "Hunks" (change blocks) in the `Diff`.

**Step 2: Diff Verification**
For each Hunk in the Diff:
* Can you find the corresponding implementation in the `Update Snippet`?
* Is the implementation identical character-for-character (ignoring standard placeholder comments)?

**Step 3: Reverse Check**
Is there anything in the `Update Snippet` that was NOT found in the `Diff`?

---

# **2. Strict Completeness and Correctness Check**

## **1. Comments count.**

Even a single updated character in a comment must appear in the Update Snippet.
If a comment in the Diff is changed but Update Snippet does not mention it → **FAIL**.

## **2. Whitespace counts.**

Modified indentation, blank lines, spacing—if shown in Diff but absent in Update Snippet → **FAIL**.

## **3. Update Snippet is the authoritative specification.**

The Diff must be fully explainable *only* based on the Update Snippet.
No inferred or guessed changes are allowed.

## **4. No creative interpretation.**

If the Update Snippet uses `// ... existing code ...`, you must treat it as meaning "no change here".
No modifications inside omitted sections are permitted.

## **5. Position verification.**
You should observe the differences and update the code snippets by changing the code near the updated code, and then determine whether the changes in the updated code snippets apply to the original code at the same location.

---

# **3. Output Format**

**Analysis:**
[Provide your reasoning here. If you find a discrepancy, describe it specifically: e.g., "Diff shows a variable rename 'x' -> 'y' on line 10, but Update Snippet retains 'x'.". Once you find a discrepancy, you should output <Judge>FAILED</Judge> and stop the analysis.]

**Verdict:**
<Judge>PASSED</Judge> OR <Judge>FAILED</Judge>""",
"user": """<original_code>{original_code}</original_code>

<final_code>{final_code}</final_code>

<update_snippet>{update_snippet}</update_snippet>

<diff>{diff}</diff>"""
}

PROMPT_TRAIN = {
  "system": """You are a deterministic Code Patching Engine. Your task is to synthesize a "Updated File" by applying a partial "Update Snippet" to the provided "Source File".

### Algorithm
1. **Context Matching**: Analyze the `Update Snippet` to identify the context anchors (the lines of code surrounding the changes). Locate the exact corresponding block in the `Source File`. The match must be unique.
2. **Code Merging**: Replace the matched block in the `Source File` with the logic from the `Update Snippet`.
3. **Expansion**: The `Update Snippet` contains omission markers (e.g., `// ... existing code ...`). You MUST replace these markers with the original, unchanged lines from the `Source File`.
4. **Output Generation**: Output the FULL content of the resulting file.

### Constraints
- **NO Laziness**: Never output comments like `// ... rest of code ...` in the final output. You must write out every single line of the final code.
- **Strict Fidelity**: Preserve the original indentation style (spaces/tabs) and comments of the Source File for all unchanged parts.
- **Safety**: If the context in the snippet is ambiguous or cannot be found, output nothing inside the tags.

### Output Format
<update_file>[Your final code here]</update_file>""",

"user": """<language>{language}</language>

<source_file>{source_file}</source_file>

<update_snippet>{update_snippet}</update_snippet>

Please generate the full updated code strictly following the instructions."""
}