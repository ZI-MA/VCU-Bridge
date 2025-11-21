import os
import openai
import json
import re
import logging

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI API client wrapper."""

    def __init__(self, api_key=None, base_url=None, model="gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY.")
        
        self.base_url = base_url or os.getenv("OPENAI_API_BASE")
        self.model = model
        self.call_count = 0
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_completion_with_custom_system(self, prompt_content, system_prompt, temperature=0.7):
        """Get completion with custom system prompt.
        
        Args:
            prompt_content: Can be a string or a list of content parts (e.g., [{"type": "text", "text": "..."}, {"type": "image_url", ...}])
            system_prompt: Custom system prompt string
            temperature: Sampling temperature
        """
        self.call_count += 1

        # Handle both string and list formats for prompt_content
        # OpenAI SDK supports both formats directly
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_content},
        ]

        try:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
            except Exception:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )

            response_text = response.choices[0].message.content
            if not response_text or not response_text.strip():
                return None
            return json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            result = self._find_json_object(response_text)
            return result if result else None
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}", exc_info=True)
            return None

    def _find_json_object(self, text):
        """Find balanced JSON object in text."""
        for i, char in enumerate(text):
            if char == '{':
                brace_count = 0
                for j, c in enumerate(text[i:], i):
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                return json.loads(text[i:j+1])
                            except json.JSONDecodeError:
                                continue
        return None

    def test_connection(self):
        """Test API connection."""
        try:
            print(f"Testing connection to {self.base_url}...")
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            print("API connection test passed.")
        except Exception as e:
            print(f"\n[Connection Test Failed]")
            print(f"Error: {e}")
            print("Check your PROXY and API KEY settings.\n")
            raise e

