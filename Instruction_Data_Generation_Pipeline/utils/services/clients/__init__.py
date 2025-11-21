"""Client factory for model backends."""

from typing import Any
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient

__all__ = ["create_client", "GeminiClient", "OpenAIClient"]


def create_client(config: Any):
    """Create a model client from config."""
    cfg = config if isinstance(config, dict) else getattr(config, "_data", {})
    client = (cfg or {}).get("client", {})
    provider = (client.get("provider") or "gemini").lower()

    if provider == "gemini":
        g = client.get("gemini", {})
        model = g.get("model") or "gemini-2.0-flash-exp"
        return GeminiClient(
            model=model,
            project_id=g.get("project_id"),
            location=g.get("location") or "us-central1",
            service_account_file=g.get("service_account_file"),
        )
    else:
        o = client.get("openai", {})
        model = o.get("model") or "gpt-4o"
        return OpenAIClient(
            api_key=o.get("api_key"),
            base_url=o.get("base_url"),
            model=model,
        )
