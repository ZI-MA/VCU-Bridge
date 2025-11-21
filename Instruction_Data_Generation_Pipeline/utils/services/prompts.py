"""Prompt loader for hierarchical reasoning generation."""

import os


_HERE = os.path.dirname(__file__)
# prompts.py is in utils/services/, so we need to go up two levels to reach Instruction_Data_Generation_Pipeline/
# then go to prompts/
PROMPTS_BASE = os.path.join(_HERE, "..", "..", "prompts")


def get_prompt(name: str, default: str = "") -> str:
    """Load a prompt by filename from prompts directory.

    Example:
    - name="qa_level1" -> prompts/qa_level1.txt
    - name="evaluation_template" -> prompts/evaluation_template.txt
    """
    filename = f"{name}.txt"
    path = os.path.join(PROMPTS_BASE, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default
