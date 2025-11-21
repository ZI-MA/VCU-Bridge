"""
QA Generator
- Externalizes long prompt templates via prompts loader
- Keeps API compatible with original generator
"""

import logging
import random
import time
import os
from typing import Dict, Optional, Any

from ..prompts import get_prompt
from ..media import encode_image_b64
from ...random import derive_int_seed


logger = logging.getLogger(__name__)


class QAGenerator:
    """Generates hierarchical reasoning nodes with question-answer pairs."""

    def __init__(self, client, config: dict = None):
        self.client = client
        self.config = config or {}
        
        # Read all parameters from config
        client_cfg = self.config.get('client', {})
        run_cfg = self.config.get('run', {})
        
        self.max_retries = int(client_cfg.get('max_retries', 3))
        self.base_delay = float(client_cfg.get('backoff_base_delay', 1.0))
        self.max_delay = float(client_cfg.get('backoff_max_delay', 30.0))
        self._seed = run_cfg.get('random_seed')
        
        # Temperature settings
        self.base_temperature = client_cfg.get("generator_temperature_base", 0.7)
        self.retry_increment = client_cfg.get("generator_temperature_retry_increment", 0.2)
        self.max_temperature = client_cfg.get("generator_temperature_max", 1.5)

        # Load templates from prompts folder
        self.templates = {
            1: get_prompt("qa_level1"),
            2: get_prompt("qa_level2"),
            3: get_prompt("qa_level3"),
        }
        # Validate that templates were loaded successfully
        for level, template in self.templates.items():
            if not template or not template.strip():
                logger.error(f"Failed to load template for level {level}. Template is empty.")
                raise ValueError(f"Template for level {level} is empty or could not be loaded.")
        
        self.system_prompt = get_prompt(
            "qa_system",
            default=(
                "You are an expert at creating hierarchical visual question-answer datasets. "
                "You MUST return ONLY valid JSON output without any extra text, explanations, markdown formatting, or code blocks. "
                "The response should start with { and end with }."
            ),
        )
        if not self.system_prompt or not self.system_prompt.strip():
            logger.warning("System prompt is empty, using default.")
            self.system_prompt = (
                "You are an expert at creating hierarchical visual question-answer datasets. "
                "You MUST return ONLY valid JSON output without any extra text, explanations, markdown formatting, or code blocks. "
                "The response should start with { and end with }."
            )

    def generate_qa_pair(self, parent_node, image_path: str = None, target_level: int = None, 
                         failure_feedback: str = None) -> Optional[Dict[str, Any]]:
        if target_level is None:
            target_level = parent_node.level + 1
        if target_level > 3:
            return None

        template = self.templates.get(target_level)
        if not template:
            return None

        context = self._build_generation_context(parent_node, target_level)
        
        # Add failure feedback for retry guidance (similar to sequential_generator)
        if failure_feedback:
            context['retry_guidance'] = f"Previous attempt failed evaluation with this reason: {failure_feedback}\nDo NOT generate similar question content, answer choices, or logical structures as the previous attempt. Approach this from a completely different angle, using different concepts, knowledge points, and reasoning pathways."
        else:
            context['retry_guidance'] = "First generation attempt."

        formatted_prompt = template.format(**context)

        prompt_content = [{"type": "text", "text": formatted_prompt}]
        if image_path and os.path.exists(image_path):
            b64 = encode_image_b64(image_path)
            if b64:
                prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            else:
                logger.warning(f"Failed to encode image {image_path}; continuing without image context")

        for attempt in range(self.max_retries):
            try:
                # Add retry guidance for internal retry attempts
                if attempt > 0:
                    context['retry_guidance'] = f"Previous attempts failed. Try a completely different approach. Use different concepts, knowledge areas, and reasoning pathways. Attempt #{attempt + 1}."
                    formatted_prompt = template.format(**context)
                    prompt_content[0] = {"type": "text", "text": formatted_prompt}

                # Calculate temperature based on retry attempt
                temperature = min(self.max_temperature, self.base_temperature + (attempt * self.retry_increment))

                response = self.client.get_completion_with_custom_system(prompt_content, self.system_prompt, temperature=temperature)
                if isinstance(response, str):
                    try:
                        import json as _json
                        response = _json.loads(response)
                    except Exception:
                        pass
                
                if response is None:
                    logger.warning(f"QA generation attempt {attempt + 1}: API returned None")
                elif not isinstance(response, dict):
                    logger.warning(f"QA generation attempt {attempt + 1}: API returned non-dict type: {type(response)}")
                elif not self._validate_qa_response(response):
                    logger.warning(f"QA generation attempt {attempt + 1}: Response validation failed. Response keys: {list(response.keys()) if isinstance(response, dict) else 'N/A'}")
                    response_str = str(response)[:200] if response else "None"
                    logger.debug(f"Invalid response content (first 200 chars): {response_str}")
                
                if response and self._validate_qa_response(response):
                    return response
                if attempt < self.max_retries - 1:
                    time.sleep(self._calculate_backoff_time(attempt))
            except KeyError as e:
                logger.error(f"QA generation attempt {attempt + 1}: Template formatting error - missing key: {e}")
                logger.error(f"Context keys available: {list(context.keys())}")
                if attempt < self.max_retries - 1:
                    time.sleep(self._calculate_backoff_time(attempt))
            except Exception as e:
                logger.warning(f"QA generation attempt {attempt + 1} failed: {e}", exc_info=True)
                if attempt < self.max_retries - 1:
                    time.sleep(self._calculate_backoff_time(attempt))
        
        logger.warning(f"QA generation failed after {self.max_retries} attempts for parent node level {parent_node.level}, target level {target_level}")
        return None

    def _build_generation_context(self, parent_node, target_level: int) -> Dict[str, str]:
        def level_desc(level: int) -> str:
            return {
                1: "Basic perception and direct observation questions about what is directly visible in the image",
                2: "Connection and relationship questions that link basic observations to broader understanding",
                3: "High-level reasoning questions that require deep inference, analysis, and abstract thinking",
            }.get(level, "Unknown level")

        def difficulty(parent_level: int, tgt: int) -> str:
            # For level 1, focus on basic perception without comparing to level 0
            if tgt == 1:
                return "Create a question appropriate for level 1 (basic perception, direct observation)."
            return (
                f"Make the question significantly more challenging than level {parent_level}, requiring deeper thinking and more complex reasoning."
                if tgt > parent_level else f"Create a question appropriate for level {tgt}."
            )

        # Base context common to all levels
        context = {
            "target_level": target_level,
            "level_description": level_desc(target_level),
            "difficulty_guidance": difficulty(parent_node.level, target_level),
        }
        # Include parent Q/A for all levels for compatibility
        # This matches the original implementation behavior
        context.update({
            "parent_question": getattr(parent_node, "question", ""),
            "parent_answer": getattr(parent_node, "answer", ""),
        })
        if getattr(parent_node, "generation_context", None):
            context.update(parent_node.generation_context)
        return context

    def _validate_qa_response(self, resp: Any) -> bool:
        """Validate that response contains non-empty string question/answer.

        Aligns with original strictness to avoid admitting null/empty values.
        """
        if not isinstance(resp, dict):
            return False
        for key in ("question", "answer"):
            if key not in resp:
                return False
            val = resp[key]
            if not isinstance(val, str):
                return False
            if not val.strip():
                return False
        return True

    def _calculate_backoff_time(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        s = derive_int_seed("qa_backoff", attempt, base=(self._seed or 0))
        r = random.Random(s)
        jitter = delay * 0.1 * (r.random() * 2 - 1)
        return max(0.1, delay + jitter)
