import os
import json
import re
import logging
from openai import OpenAI
from google.oauth2 import service_account
from google.auth.transport.requests import Request

logger = logging.getLogger(__name__)


class GeminiClient:
    """Gemini API client using OpenAI-compatible interface."""

    def __init__(self, model="gemini-2.0-flash-exp", project_id=None, 
                 location="us-central1", service_account_file=None):
        self.model = model
        self.project_id = project_id or os.getenv("GEMINI_PROJECT_ID")
        self.location = location or os.getenv("GEMINI_LOCATION", "us-central1")
        self.call_count = 0
        
        if not self.project_id:
            raise ValueError("project_id required. Set GEMINI_PROJECT_ID.")
        
        if service_account_file is None:
            service_account_file = os.getenv("GEMINI_SERVICE_ACCOUNT_FILE")
        
        if not service_account_file:
            raise ValueError("service_account_file required. Set GEMINI_SERVICE_ACCOUNT_FILE.")

        try:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            credentials.refresh(Request())
            
            api_host = f"{self.location}-aiplatform.googleapis.com" if self.location != "global" else "aiplatform.googleapis.com"
            
            self.client = OpenAI(
                base_url=f"https://{api_host}/v1/projects/{self.project_id}/locations/{self.location}/endpoints/openapi",
                api_key=credentials.token,
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {e}")

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
            {"role": "user", "content": prompt_content}
        ]

        try:
            try:
                response = self.client.chat.completions.create(
                    model=f"google/{self.model}",
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
            except Exception:
                response = self.client.chat.completions.create(
                    model=f"google/{self.model}",
                    messages=messages,
                    temperature=temperature,
                )
            
            response_text = response.choices[0].message.content
            if not response_text or not response_text.strip():
                return None
            return json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            result = self._find_json_object(response_text)
            return result if result else None
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}", exc_info=True)
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
        print(f"Testing connection to {self.client.base_url}...")
        try:
            self.client.chat.completions.create(
                model=f"google/{self.model}",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            print("API connection test passed.")
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

