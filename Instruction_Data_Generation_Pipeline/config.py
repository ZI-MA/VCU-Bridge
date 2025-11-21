import json
import logging
import os
from typing import Any, Dict


class RuntimeConfig:
    """Runtime config loader with environment variable override support."""

    def __init__(self, path: str):
        self._data: Dict[str, Any] = {}
        with open(path, "r", encoding="utf-8") as f:
            self._data.update(json.load(f))
        self._apply_env()

    def _apply_env(self):
        """Override config with environment variables."""
        env = os.environ
        client = self._data.setdefault("client", {})
        
        # OpenAI
        if env.get("OPENAI_API_KEY"):
            client.setdefault("openai", {})["api_key"] = env.get("OPENAI_API_KEY")
        if env.get("OPENAI_BASE_URL"):
            client.setdefault("openai", {})["base_url"] = env.get("OPENAI_BASE_URL")
        
        # Gemini
        if env.get("GEMINI_PROJECT_ID"):
            client.setdefault("gemini", {})["project_id"] = env.get("GEMINI_PROJECT_ID")
        if env.get("GEMINI_LOCATION"):
            client.setdefault("gemini", {})["location"] = env.get("GEMINI_LOCATION")
        if env.get("GEMINI_SERVICE_ACCOUNT_FILE"):
            client.setdefault("gemini", {})["service_account_file"] = env.get("GEMINI_SERVICE_ACCOUNT_FILE")
        
        # MCTS
        mcts = self._data.setdefault("mcts", {})
        if env.get("MCTS_MAX_ITERATIONS"):
            try:
                mcts["max_iterations"] = int(env.get("MCTS_MAX_ITERATIONS"))
            except Exception:
                pass
        if env.get("MCTS_MAX_DEPTH"):
            try:
                mcts["max_depth"] = int(env.get("MCTS_MAX_DEPTH"))
            except Exception:
                pass

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration."""
    level = getattr(logging, str(config.get("log_level", "INFO")).upper(), logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handlers = []
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    handlers.append(ch)
    
    log_file = config.get("log_file")
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        handlers.append(fh)
    
    logging.basicConfig(level=level, handlers=handlers, force=True)
    
    # Reduce noise from external libraries
    for ln in ['openai', 'openai._base_client', 'httpx', 'httpcore', 'urllib3', 'requests']:
        logging.getLogger(ln).setLevel(logging.WARNING)

