#!/usr/bin/env python3
"""Enhanced Callback Handler for OpenRouter and Cost Tracking.

This module provides an enhanced version of Langfuse's CallbackHandler that:
1. Extracts model names from additional locations (like OpenRouter's llm_output.model_name)
2. Includes cost tracking functionality similar to the langfuse.openai integration
3. Extracts OpenRouter-specific metadata (BYOK status, system fingerprint, etc.)

Usage:
    from callback_handler_patch import CostTrackingCallbackHandler

    handler = CostTrackingCallbackHandler()
    # Use with LangChain as normal, but now with enhanced model extraction and cost tracking
"""

from typing import Any, Dict, Optional, Union
from uuid import UUID

from langfuse.langchain import CallbackHandler
from langfuse.logger import langfuse_logger


class CostTrackingCallbackHandler(CallbackHandler):
    """Enhanced Langfuse CallbackHandler with OpenRouter cost tracking and metadata extraction.

    This class extends the standard Langfuse CallbackHandler by overriding _detach_observation
    to inject OpenRouter-specific cost details and metadata before the generation is finalized.

    This approach allows us to:
    - Use super().on_llm_end() for all standard processing
    - Add our enhancements at the right moment before .end() is called
    - Maintain full compatibility with the parent's logic
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._openrouter_warning_logged = False
        self._current_response = None  # Store response for _detach_observation

    def _log_openrouter_parse_warning(self, error: Exception) -> None:
        """Log OpenRouter parsing warnings only once per instance."""
        if not self._openrouter_warning_logged:
            langfuse_logger.warning(
                "Failed to parse OpenRouter data from LLM response: %s. "
                "This warning will only be shown once per handler instance.",
                error
            )
            self._openrouter_warning_logged = True

    def _parse_openrouter_data(self, response: Any) -> Dict[str, Any]:
        """Parse OpenRouter-specific cost and metadata from LLM response.

        Args:
            response: The LLM response object

        Returns:
            Dictionary with cost_details and metadata to add to generation.update()
        """
        try:
            if not (hasattr(response, 'llm_output') and response.llm_output):
                return {}

            llm_output = response.llm_output
            if not isinstance(llm_output, dict):
                return {}

            update_params = {}

            # Parse cost information from token_usage
            if 'token_usage' in llm_output and isinstance(llm_output['token_usage'], dict):
                token_usage = llm_output['token_usage']

                # Extract cost details if available
                if 'cost' in token_usage and isinstance(token_usage['cost'], (int, float)):
                    cost = token_usage['cost']
                    if cost > 0:
                        cost_details_data = token_usage.get('cost_details', {})
                        update_params["cost_details"] = {
                            "input": cost_details_data.get("upstream_inference_prompt_cost", 0),
                            "output": cost_details_data.get("upstream_inference_completions_cost", 0),
                            "total": cost
                        }

                # Build metadata
                metadata = {}
                if 'is_byok' in token_usage:
                    metadata["openrouter_is_byok"] = token_usage['is_byok']

                # Extract other OpenRouter metadata fields
                openrouter_fields = ['system_fingerprint', 'service_tier', 'id']
                for field in openrouter_fields:
                    if field in llm_output and llm_output[field] is not None:
                        metadata[f"openrouter_{field}"] = llm_output[field]

                if metadata:
                    update_params["metadata"] = metadata

            return update_params

        except Exception as e:
            self._log_openrouter_parse_warning(e)
            return {}

    def _detach_observation(self, run_id: UUID) -> Optional[Union[Any]]:
        """Override to inject OpenRouter data before parent finalizes the generation."""
        # Get the generation before parent detaches it
        generation = super()._detach_observation(run_id)

        # If we have a generation and stored response, add OpenRouter data
        if generation is not None and self._current_response is not None:
            openrouter_params = self._parse_openrouter_data(self._current_response)
            if openrouter_params:
                # Update with OpenRouter data before parent calls .end()
                generation.update(**openrouter_params)

        return generation

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Store response and delegate to parent for all processing."""
        # Store response for _detach_observation to use
        self._current_response = response

        try:
            # Let parent handle everything - our _detach_observation will inject OpenRouter data
            return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        finally:
            # Clean up stored response
            self._current_response = None
