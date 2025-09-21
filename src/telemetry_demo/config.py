import os
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class OpenRouterConfig:
    """OpenRouter connection settings"""
    api_key: str
    base_url: str
    site_url: str
    site_name: str
    default_model: str = "mistralai/ministral-3b"

    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            site_url=os.getenv("YOUR_SITE_URL", "https://lvh.me"),
            site_name=os.getenv("YOUR_SITE_NAME", "Your Site Name")
        )


@dataclass
class TelemetryContext:
    """Essential telemetry data for tracing"""
    session_id: str
    tags: List[str]
    metadata: Dict[str, Any]

