"""
QA Evaluator
- Externalizes long evaluation prompt via prompts loader
"""

import logging
import os
import time
import random
from typing import Dict, Any

from ..prompts import get_prompt
from ..media import encode_image_b64


logger = logging.getLogger(__name__)


class QAEvaluator:
    def __init__(self, client, config: dict = None):
        self.client = client
        self.config = config or {}
        
        # Read parameters from config
        client_cfg = self.config.get("client", {})
        self.max_retries = int(client_cfg.get('max_retries', 3))
        self.base_delay = float(client_cfg.get("backoff_base_delay", 1.0))
        self.max_delay = float(client_cfg.get("backoff_max_delay", 30.0))
        self.evaluator_temperature = float(client_cfg.get("evaluator_temperature", 0.2))
        self.fallback_quality_score = float(client_cfg.get("fallback_quality_score", 0.5))
        
        self.template = get_prompt("evaluation_template")
        self.system_prompt = get_prompt(
            "eval_system",
            default=(
                "You are a strict evaluator for hierarchical visual Q&A datasets. "
                "You MUST return ONLY valid JSON (the response must start with '{' and end with '}'), "
                "with no extra text, explanations, markdown, or code fences. "
                "Return {\"quality_score\": 0.0-1.0, \"reasoning\": \"brief evaluation summary including strengths and areas for improvement\"}."
            ),
        )

    def evaluate_qa_quality(self, child_node, parent_node, image_path: str = None) -> tuple[float, str]:
        context = self._build_context(child_node, parent_node)
        formatted = self.template.format(**context)
        prompt_content = [{"type": "text", "text": formatted}]
        if image_path and os.path.exists(image_path):
            b64 = encode_image_b64(image_path)
            if b64:
                prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            else:
                logger.warning(f"Failed to encode image {image_path}; continuing without image context")

        for attempt in range(self.max_retries):
            try:
                # Get temperature from new client config
                temperature = self.evaluator_temperature
                response = self.client.get_completion_with_custom_system(prompt_content, self.system_prompt, temperature=temperature)
                if isinstance(response, str):
                    try:
                        import json as _json
                        response = _json.loads(response)
                    except Exception:
                        pass
                if response and self._validate_response(response):
                    return self._evaluate_quality(response)
            except Exception as e:
                logger.warning(f"QA evaluation attempt {attempt + 1} failed: {e}")
            # Backoff before next retry when applicable
            if attempt < self.max_retries - 1:
                time.sleep(self._calculate_backoff_time(attempt))
        # Fallback: return fallback score with reason
        return self.fallback_quality_score, "Evaluation API call failed, using fallback score"

    def _build_context(self, child_node, parent_node) -> Dict[str, str]:
        def level_desc(level: int) -> str:
            return {
                1: "basic perception and direct observation",
                2: "connections and relationships between concepts", 
                3: "high-level reasoning and abstract analysis",
            }.get(level, "unknown level")

        return {
            "parent_question": parent_node.question,
            "parent_answer": parent_node.answer,
            "parent_level": str(parent_node.level),
            "child_question": child_node.question,
            "child_answer": child_node.answer,
            "child_level": str(child_node.level),
            "level_description": level_desc(child_node.level),
        }

    def _validate_response(self, response: Dict[str, Any]) -> bool:
        if not isinstance(response, dict):
            return False
        if "quality_score" not in response:
            return False
        try:
            score = float(response["quality_score"])
            if not (0.0 <= score <= 1.0):
                return False
        except (ValueError, TypeError):
            return False
        if "reasoning" not in response or not isinstance(response["reasoning"], str):
            return False
        return True

    def _evaluate_quality(self, response: Dict[str, Any]) -> tuple[float, str]:
        quality_score = float(response.get("quality_score", 0.0))
        reasoning = response.get("reasoning", "No Reasoning")
        
        # Ensure quality score is in valid range
        quality_score = max(0.0, min(1.0, quality_score))
        
        return quality_score, reasoning

    def _calculate_backoff_time(self, attempt: int) -> float:
        # Exponential backoff with light jitter
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        # Deterministic-ish jitter based on attempt
        rnd = random.random()
        jitter = delay * 0.1 * (rnd * 2 - 1)
        return max(0.1, delay + jitter)
