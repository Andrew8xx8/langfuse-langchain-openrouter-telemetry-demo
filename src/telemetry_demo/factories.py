from datetime import datetime
from typing import List, Dict, Any

from langfuse.openai import openai
from langchain_openai import ChatOpenAI
from .callback_handler_patch import CostTrackingCallbackHandler

from .config import OpenRouterConfig, TelemetryContext


class LLMClientFactory:
    """Creates LLM clients from configuration"""

    @staticmethod
    def create_openai_client(config: OpenRouterConfig) -> openai.OpenAI:
        return openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers={
                "HTTP-Referer": config.site_url,
                "X-Title": config.site_name,
            }
        )

    @staticmethod
    def create_langchain_llm(config: OpenRouterConfig) -> ChatOpenAI:
        return ChatOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.default_model,
            default_headers={
                "HTTP-Referer": config.site_url,
                "X-Title": config.site_name,
            },
            model_kwargs={"extra_body": {"usage": {"include": True}}}
        )


class TelemetryFactory:
    """Creates telemetry components and contexts"""

    @staticmethod
    def create_handler() -> CostTrackingCallbackHandler:
        return CostTrackingCallbackHandler()

    @staticmethod
    def create_session_id(prefix: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}-{timestamp}"

    @staticmethod
    def create_context(session_id: str, base_tags: List[str], **metadata) -> TelemetryContext:
        return TelemetryContext(
            session_id=session_id,
            tags=base_tags,
            metadata={
                "langfuse_session_id": session_id,
                "langfuse_tags": base_tags,
                "environment": "demo",
                **metadata
            }
        )

