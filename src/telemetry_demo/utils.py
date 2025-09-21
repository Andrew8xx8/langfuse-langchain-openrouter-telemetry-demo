from typing import Dict, Any, Optional

from .config import TelemetryContext


class CostExtractor:
    """Extracts cost information from responses"""

    @staticmethod
    def extract_cost_info(response_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        usage = response_metadata.get("usage") or response_metadata.get("token_usage") or {}
        cost = usage.get("cost")
        cost_details = usage.get("cost_details")

        if cost:
            return {
                "cost": cost,
                "cost_details": cost_details
            }
        return None

    @staticmethod
    def display_cost_info(cost_info: Optional[Dict[str, Any]], context: str = ""):
        if cost_info:
            cost = cost_info["cost"]
            cost_details = cost_info["cost_details"]
            print(f"OpenRouter cost{' ' + context if context else ''}: ${cost}")
            if cost_details:
                print(f"Cost breakdown: {cost_details}")
            print("✓ Cost data automatically forwarded to Langfuse")
        else:
            print("No cost data found in response")


class MetadataBuilder:
    """Builds metadata dictionaries for different contexts"""

    @staticmethod
    def for_openai(telemetry_context: TelemetryContext, **extras) -> Dict[str, Any]:
        return {**telemetry_context.metadata, **extras}

    @staticmethod
    def for_langchain_config(telemetry_context: TelemetryContext, **extras) -> Dict[str, Any]:
        return {
            "metadata": {**telemetry_context.metadata, **extras}
        }

