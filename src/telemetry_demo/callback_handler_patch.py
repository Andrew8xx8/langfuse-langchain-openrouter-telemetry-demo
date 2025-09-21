#!/usr/bin/env python3
"""
Cost-Tracking Callback Handler Patch

This module provides a patched version of Langfuse's CallbackHandler that includes
OpenRouter cost tracking functionality. It inherits from the original CallbackHandler
and adds cost parsing similar to the langfuse.openai integration.

Usage:
    from callback_handler_patch import CostTrackingCallbackHandler

    handler = CostTrackingCallbackHandler()
    # Use with LangChain as normal, but now with cost tracking
"""

from typing import Any, Dict, Optional
from uuid import UUID
from langfuse.langchain import CallbackHandler
from langfuse.logger import langfuse_logger


class CostTrackingCallbackHandler(CallbackHandler):
    """
    Enhanced Langfuse CallbackHandler that includes OpenRouter cost tracking.

    This class inherits from the standard Langfuse CallbackHandler and adds
    cost parsing functionality similar to the langfuse.openai integration.
    It automatically extracts and forwards cost information from OpenRouter
    and other providers that include cost data in their responses.
    """

    def __init__(self, *, public_key: Optional[str] = None):
        """Initialize the CostTrackingCallbackHandler.

        Args:
            public_key: Optional Langfuse public key. If not provided, will use default client.
            update_trace: Deprecated/unused parameter.
        """
        super().__init__(public_key=public_key)
        self._cost_parse_warning_logged = False

    def _parse_cost_from_llm_result(self, response: Any) -> Optional[Dict[str, Any]]:
        """
        Parse cost information from LangChain LLMResult.

        This method extracts cost data from various locations in the LLM response,
        supporting OpenRouter and other providers that include cost information.

        Args:
            response: LangChain LLMResult object

        Returns:
            Dict with cost details in Langfuse format, or None if no cost found
        """
        try:
            # Method 1: Check llm_output for usage/token_usage with cost
            if hasattr(response, 'llm_output') and response.llm_output:
                for usage_key in ["usage", "token_usage"]:
                    if usage_key in response.llm_output:
                        usage_data = response.llm_output[usage_key]

                        # Handle dict format
                        if isinstance(usage_data, dict) and "cost" in usage_data:
                            cost = usage_data["cost"]
                            if isinstance(cost, (int, float)) and cost > 0:
                                cost_details = usage_data.get("cost_details", {})
                                return {
                                    "input": cost_details.get("upstream_inference_prompt_cost", 0),
                                    "output": cost_details.get("upstream_inference_completions_cost", 0),
                                    "total": cost
                                }

                        # Handle object format
                        elif hasattr(usage_data, "cost"):
                            cost = getattr(usage_data, "cost")
                            if isinstance(cost, (int, float)) and cost > 0:
                                return {"total": cost}

            # Method 2: Check generations for response_metadata with cost
            if hasattr(response, 'generations') and response.generations:
                for generation_list in response.generations:
                    for generation in generation_list:
                        # Check message response_metadata
                        if hasattr(generation, 'message'):
                            message = generation.message
                            if hasattr(message, 'response_metadata'):
                                meta = message.response_metadata or {}
                                usage = meta.get("usage") or meta.get("token_usage") or {}

                                if "cost" in usage:
                                    cost = usage["cost"]
                                    if isinstance(cost, (int, float)) and cost > 0:
                                        cost_details = usage.get("cost_details", {})
                                        return {
                                            "input": cost_details.get("upstream_inference_prompt_cost", 0),
                                            "output": cost_details.get("upstream_inference_completions_cost", 0),
                                            "total": cost
                                        }

                        # Check generation_info for usage_metadata
                        if hasattr(generation, 'generation_info') and generation.generation_info:
                            gen_info = generation.generation_info
                            if "usage_metadata" in gen_info:
                                usage_meta = gen_info["usage_metadata"]
                                if isinstance(usage_meta, dict) and "cost" in usage_meta:
                                    cost = usage_meta["cost"]
                                    if isinstance(cost, (int, float)) and cost > 0:
                                        return {"total": cost}
                                elif hasattr(usage_meta, "cost"):
                                    cost = getattr(usage_meta, "cost")
                                    if isinstance(cost, (int, float)) and cost > 0:
                                        return {"total": cost}

            return None

        except Exception as e:
            if not self._cost_parse_warning_logged:
                langfuse_logger.warning("Failed to parse cost information: %s", e)
                self._cost_parse_warning_logged = True
            return None

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Enhanced on_llm_end that includes cost tracking.

        This method extends the parent's on_llm_end to include cost parsing
        and forwarding to Langfuse, similar to the langfuse.openai integration.
        """
        try:
            self._log_debug_event(
                "on_llm_end", run_id, parent_run_id, response=response, kwargs=kwargs
            )

            # Extract response content (same as parent)
            response_generation = response.generations[-1][-1]
            extracted_response = (
                self._convert_message_to_dict(response_generation.message)
                if hasattr(response_generation, 'message') and response_generation.message
                else str(response_generation.text) if hasattr(response_generation, 'text') else str(response_generation)
            )

            # Parse usage and model (same as parent)
            from langfuse.langchain.CallbackHandler import _parse_usage, _parse_model
            llm_usage = _parse_usage(response)
            model = _parse_model(response)

            # NEW: Parse cost information
            cost_details = self._parse_cost_from_llm_result(response)

            generation = self._detach_observation(run_id)

            if generation is not None:
                update_params = {
                    "output": extracted_response,
                    "usage": llm_usage,  # backward compat
                    "usage_details": llm_usage,
                    "input": kwargs.get("inputs"),
                    "model": model,
                }

                # Add cost details if available
                if cost_details:
                    update_params["cost_details"] = cost_details
                    langfuse_logger.debug("Added cost details to generation %s: %s", run_id, cost_details)

                generation.update(**update_params).end()

        except Exception as e:
            langfuse_logger.exception(e)

        finally:
            # Clean up completion start time memo (same as parent)
            if hasattr(self, 'updated_completion_start_time_memo'):
                self.updated_completion_start_time_memo.discard(run_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Enhanced on_llm_error that includes cost details in error cases.

        This ensures consistent cost_details formatting even in error scenarios.
        """
        try:
            self._log_debug_event("on_llm_error", run_id, parent_run_id, error=error)

            generation = self._detach_observation(run_id)

            if generation is not None:
                generation.update(
                    status_message=str(error),
                    level="ERROR",
                    input=kwargs.get("inputs"),
                    cost_details={"input": 0, "output": 0, "total": 0},  # Consistent with openai.py
                ).end()

        except Exception as e:
            langfuse_logger.exception(e)

