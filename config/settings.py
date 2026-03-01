import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Settings:
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    ollama_remote_api_key: Optional[str] = None
    ollama_remote_url: Optional[str] = None
    
    screenshot_folder: Path = field(default_factory=lambda: Path("data/screenshots"))
    reference_pattern: str = "reference"
    comparison_pattern: str = "comparison"
    
    export_columns: list[str] = field(default_factory=lambda: [
        "pair_id", "mllm", "element", "presence_a", "presence_b", "match"
    ])
    optional_columns: list[str] = field(default_factory=lambda: [
        "reasoning", "tokens_used", "latency_ms", "raw_response", "timestamp"
    ])
    
    def __post_init__(self):
        load_dotenv()
        
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.google_api_key is None:
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if self.ollama_remote_api_key is None:
            self.ollama_remote_api_key = os.getenv("OLLAMA_REMOTE_API_KEY")
        if self.ollama_remote_url is None:
            self.ollama_remote_url = os.getenv("OLLAMA_REMOTE_URL")
    
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)
    
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)
    
    def has_google(self) -> bool:
        return bool(self.google_api_key)
    
    def has_ollama_remote(self) -> bool:
        return bool(self.ollama_remote_api_key and self.ollama_remote_url)
    
    def available_providers(self) -> list[str]:
        providers = []
        if self.has_openai():
            providers.append("GPT-4V")
        if self.has_anthropic():
            providers.append("Claude Vision")
        if self.has_google():
            providers.append("Gemini Pro")
        return providers


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def update_settings(**kwargs) -> Settings:
    global _settings
    _settings = Settings(**kwargs)
    return _settings
