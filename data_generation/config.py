"""
aiXapply data generation pipeline.

Stage: shared configuration.
Purpose: define model names and generation parameters for each LLM stage.
Inputs: environment variable overrides and prompt templates.
Outputs: BATCH_REQUEST_CONFIG used by request dispatch scripts.
"""
import os

from prompt import (
    PROMPT_CODE_GENERATION,
    PROMPT_DESCRIPTION,
    PROMPT_VERIFICATION,
)

BATCH_REQUEST_CONFIG = {
    "description_generation": {
        "model": os.getenv("DESCRIPTION_MODEL", "deepseek-v3-2-251201"),
        "temperature": 0.3,
        "top_p": 0.9,
        "penalty": 0,
        "system_prompt": PROMPT_DESCRIPTION['system'],
        "query_type": "description",
    },
    "code_generation": {
        "model": os.getenv("CODE_GENERATION_MODEL", "claude-opus-4-5-20251101"),
        "temperature": 0.3,
        "top_p": 0.9,
        "penalty": 0,
        "system_prompt": PROMPT_CODE_GENERATION['system'],
        "query_type": "code",
    },
    "verification": {
        "model": os.getenv("VERIFICATION_MODEL", "deepseek-v3-2-251201"),
        "temperature": 0.3,
        "top_p": 0.9,
        "penalty": 0,
        "system_prompt": PROMPT_VERIFICATION['system'],
        "query_type": "verification",
    }
}

