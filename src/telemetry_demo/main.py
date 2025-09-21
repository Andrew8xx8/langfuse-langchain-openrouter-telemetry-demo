#!/usr/bin/env python3
"""
Clean Telemetry Demo - Refactored architecture

Demonstrates cost tracking with clean separation of concerns:
- Configuration: Pure data classes
- Factories: Object creation utilities
- Utils: Single-purpose helpers
- Runner: Test orchestration
"""

import os
from dotenv import load_dotenv

from .config import OpenRouterConfig
from .factories import TelemetryFactory
from .runner import TestRunner


def print_environment_info():
    """Print current environment configuration"""
    print("=== Environment Configuration ===")
    print(f"OPENROUTER_API_KEY: {'✓ Set' if os.getenv('OPENROUTER_API_KEY') else '✗ Not set'}")
    print(f"OPENROUTER_BASE_URL: {os.getenv('OPENROUTER_BASE_URL', 'Not set (using default)')}")
    print(f"LANGFUSE_PUBLIC_KEY: {'✓ Set' if os.getenv('LANGFUSE_PUBLIC_KEY') else '✗ Not set'}")
    print(f"LANGFUSE_SECRET_KEY: {'✓ Set' if os.getenv('LANGFUSE_SECRET_KEY') else '✗ Not set'}")
    print(f"LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST', 'Not set (using default)')}")
    print()


def main():
    """Application entry point"""
    load_dotenv()

    print("🧪 Clean Telemetry Demo - OpenRouter + Langfuse Cost Tracking\n")
    print_environment_info()

    try:
        # Configuration
        config = OpenRouterConfig.from_env()
        runner = TestRunner(config)

        # Create telemetry contexts for each test
        direct_context = TelemetryFactory.create_context(
            session_id=TelemetryFactory.create_session_id("direct-openai"),
            base_tags=["cost-tracking", "direct", "openai"]
        )

        langchain_context = TelemetryFactory.create_context(
            session_id=TelemetryFactory.create_session_id("langchain-cost-tracking"),
            base_tags=["cost-tracking", "langchain", "patch"]
        )

        langgraph_context = TelemetryFactory.create_context(
            session_id=TelemetryFactory.create_session_id("langgraph-cost-tracking"),
            base_tags=["cost-tracking", "langgraph", "workflow"]
        )

        # Run tests
        runner.run_direct_openai_test(direct_context)
        runner.run_langchain_test(langchain_context)
        runner.run_langgraph_test(langgraph_context)

        print("Summary:")
        print("1. Direct OpenAI: Uses langfuse.openai wrapper")
        print("2. LangChain with Cost Tracking: Uses CostTrackingCallbackHandler patch")
        print("3. LangGraph Multi-Node: Uses CostTrackingCallbackHandler with workflow")
        print()
        print("Notes:")
        print("- OpenRouter returns cost information when extra_body={'usage': {'include': True}}")
        print("- All approaches properly track and forward cost data to Langfuse")
        print("- Clean architecture separates configuration, creation, and execution")
        print("- Check Langfuse dashboard to verify cost tracking across all approaches")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

