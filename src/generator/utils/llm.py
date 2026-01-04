from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from ..core.config import GeneratorConfig

# LangChain providers supported by init_chat_model without additional routing.
KNOWN_PROVIDERS = {
    "openai",
    "anthropic",
    "azure_openai",
    "google_genai",
    "google_vertexai",
    "bedrock",
    "bedrock_converse",
    "cohere",
    "fireworks",
    "together",
    "mistralai",
    "huggingface",
    "groq",
    "deepseek",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_llm_kwargs(
    config: GeneratorConfig,
    logger: logging.Logger,
    *,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    temperature_override: Optional[float] = None,
    seed_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare kwargs for langchain.chat_models.init_chat_model based on the configuration.

    Unknown providers are automatically routed through OpenRouter using an OpenAI-compatible API.

    Args:
        config: GeneratorConfig instance supplying defaults.
        logger: Logger used for informational output.
        provider_override: Optional provider override for the LLM call.
        model_override: Optional model override for the LLM call.
        temperature_override: Optional temperature override (defaults to config.temperature).
        seed_override: Optional seed override (defaults to config.seed when supported).

    Returns:
        Dictionary of kwargs suitable for init_chat_model.

    Raises:
        ValueError: If routing through OpenRouter is required but the API key is missing.
    """

    provider = provider_override or config.model_provider
    model_name = model_override or config.model_name
    temperature = temperature_override if temperature_override is not None else config.temperature
    seed = seed_override if seed_override is not None else config.seed

    if provider not in KNOWN_PROVIDERS:
        logger.info("Provider '%s' not in known providers list. Routing through OpenRouter.", provider)

        openrouter_model = f"{provider}/{model_name}"
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                f"Unknown provider '{provider}' requires OpenRouter. "
                "Please set OPENROUTER_API_KEY environment variable."
            )

        llm_kwargs: Dict[str, Any] = {
            "model": openrouter_model,
            "model_provider": "openai",  # OpenRouter uses the OpenAI-compatible interface
            "temperature": temperature,
            "base_url": OPENROUTER_BASE_URL,
            "api_key": api_key,
        }

        logger.info("Using OpenRouter with model: %s", openrouter_model)
    else:
        llm_kwargs = {
            "model": model_name,
            "model_provider": provider,
            "temperature": temperature,
        }

        if seed is not None:
            # Google GenAI requires seed in model_kwargs, not as direct parameter
            if provider == "google_genai":
                llm_kwargs["model_kwargs"] = {"seed": seed}
            else:
                llm_kwargs["seed"] = seed

    return llm_kwargs
