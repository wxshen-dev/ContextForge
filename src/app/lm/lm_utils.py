# Environment configuration and dependencies.
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException
from typing import Optional

# Project dependencies.
from app.conf.lm_config import lm_config
from app.core.logger import logger

# Global cache: key is (model_name, json_mode), value is a ChatOpenAI instance.
# This avoids repeated client initialization and centralizes instance management.
_llm_client_cache = {}


def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """
    Get a globally cached LangChain ChatOpenAI client instance.
    Supports OpenAI-compatible APIs, custom models, and JSON output mode.

    :param model: Model name. Priority: argument > lm_config.llm_model > qwen3-32b.
    :param json_mode: Whether to enable JSON output mode.
    :return: Initialized ChatOpenAI instance, loaded from cache when possible.
    :raise ValueError: Missing required API key or base URL.
    :raise Exception: Model initialization failure from the LangChain layer.
    """
    # 1. Resolve the target model.
    target_model = model or lm_config.llm_model or "qwen3-32b"
    # Cache key: model name + JSON mode uniquely identify a configured client.
    cache_key = (target_model, json_mode)

    # 2. Cache hit: return the initialized instance directly.
    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM client] Cache hit. Returning instance: model={target_model}, json_mode={json_mode}")
        return _llm_client_cache[cache_key]

    # 3. Validate required API configuration early.
    if not lm_config.api_key:
        raise ValueError("[LLM client] Missing configuration: set OPENAI_API_KEY in .env.")
    if not lm_config.base_url:
        raise ValueError("[LLM client] Missing configuration: set OPENAI_API_BASE in .env.")
    logger.info(f"[LLM client] Initializing new instance: model={target_model}, json_mode={json_mode}")

    # 4. Assemble provider-specific parameters and OpenAI-compatible parameters.
    # extra_body is passed through by LangChain to compatible provider APIs.
    extra_body = {"enable_thinking": False}  # Qwen-specific: disable thinking-chain output.
    # model_kwargs contains standard OpenAI-compatible parameters.
    model_kwargs = {}
    if json_mode:
        # Force the model to return a parseable json_object.
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug("[LLM client] JSON output mode enabled; the model will return a standard JSON object.")

    # 5. Initialize the client and wrap LangChain-layer errors.
    try:
        llm_client = ChatOpenAI(
            model=target_model,  # Target model name.
            temperature=lm_config.llm_temperature or 0.1,  # Low temperature for deterministic output.
            api_key=lm_config.api_key,  # API key.
            base_url=lm_config.base_url,  # API base URL.
            extra_body=extra_body,  # Provider-specific parameters.
            model_kwargs=model_kwargs,  # OpenAI-compatible parameters.
        )
    except LangChainException as e:
        raise Exception(f"[LLM client] Model '{target_model}' initialization failed at the LangChain layer: {str(e)}") from e

    # 6. Store the new instance in the global cache.
    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM client] Instance initialized and cached: model={target_model}, json_mode={json_mode}")

    return llm_client


# Test example: verify client creation, caching, and logging.
if __name__ == "__main__":
    logger.info("===== Starting LLM client utility test =====")
    try:
        # Test 1: default configuration.
        client1 = get_llm_client()
        logger.info("Test 1 passed: default client created successfully.")

        # Test 2: specified multimodal model.
        client2 = get_llm_client(model="qwen-vl-plus")
        logger.info("Test 2 passed: specified multimodal-model client created successfully.")

        # Test 3: same model and mode to verify cache hit.
        client3 = get_llm_client(model="qwen-vl-plus")
        logger.info(f"Test 3 passed: cache verified; client2 and client3 are the same instance: {client2 is client3}")

        # Test 4: JSON output mode.
        client4 = get_llm_client(model="qwen3-32b", json_mode=True)
        logger.info("Test 4 passed: JSON-mode client created successfully.")

    except Exception as e:
        logger.error(f"LLM client utility test failed: {str(e)}", exc_info=True)
    finally:
        logger.info("===== LLM client utility test finished =====")
