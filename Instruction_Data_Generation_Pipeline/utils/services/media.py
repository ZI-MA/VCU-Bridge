"""
Media utilities shared across generator/evaluator/initialization.
"""

import base64
from typing import Optional


def encode_image_b64(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

