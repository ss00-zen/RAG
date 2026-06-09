"""
LiteLLM Gateway - HTTP Proxy Mode

This module makes HTTP requests to a LiteLLM proxy server instead of using the SDK directly.

Environment variables:
- LITELLM_PROXY_URL: Proxy URL (default: http://localhost:4000)
- LITELLM_MODEL: Model alias exposed by LiteLLM proxy (default: nvidia-llm)
- LITELLM_MASTER_KEY: LiteLLM proxy master key (required by app)
"""

import os
import requests
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from rag.logger import logger

load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if value else default.strip()


def _get_proxy_url() -> str:
    """Get LiteLLM proxy base URL."""
    return _env("LITELLM_PROXY_URL", "http://localhost:4000").rstrip("/")


def _get_master_key() -> str:
    """Get LiteLLM master key from environment."""
    master_key = _env("LITELLM_MASTER_KEY")
    if not master_key:
        raise EnvironmentError(
            "LITELLM_MASTER_KEY environment variable is not set or is empty. "
            "Please set a valid LiteLLM master key."
        )
    return master_key


def _get_auth_headers() -> Dict[str, str]:
    """Build authenticated headers for LiteLLM proxy."""
    return {
        "Authorization": f"Bearer {_get_master_key()}",
        "Content-Type": "application/json",
    }


def _check_proxy_health() -> bool:
    """
    Verify LiteLLM proxy is running and reachable.

    Uses /v1/models because proxy auth is enabled and /health without auth returns 401.
    """
    proxy_url = _get_proxy_url()
    health_url = f"{proxy_url}/v1/models"

    try:
        response = requests.get(
            health_url,
            headers=_get_auth_headers(),
            timeout=5,
        )
        is_healthy = response.status_code == 200

        if is_healthy:
            logger.debug("LiteLLM proxy is healthy at %s", proxy_url)
        else:
            logger.warning(
                "LiteLLM proxy returned status %d at %s",
                response.status_code,
                health_url,
            )

        return is_healthy

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to LiteLLM proxy at %s. Is it running?", proxy_url)
        logger.info(
            "Start the proxy with: litellm --config litellm_config.yaml --detailed_debug"
        )
        return False
    except Exception as e:
        logger.error("Error checking proxy health at %s: %s", health_url, str(e))
        return False


def call_litellm(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    **kwargs,
) -> str:
    """
    Call LLM via LiteLLM HTTP proxy.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: LiteLLM model alias (default: nvidia-llm)
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        **kwargs: Additional OpenAI-compatible params

    Returns:
        Response text from LLM

    Raises:
        EnvironmentError: If LITELLM_MASTER_KEY is not set
        RuntimeError: If proxy is unavailable or request fails
    """
    proxy_url = _get_proxy_url()
    headers = _get_auth_headers()

    if not _check_proxy_health():
        raise RuntimeError(
            f"LiteLLM proxy is not running at {proxy_url}. "
            "Please start it with: litellm --config litellm_config.yaml --detailed_debug"
        )

    model_name = model or _env("LITELLM_MODEL", "nvidia-llm")
    api_url = f"{proxy_url}/v1/chat/completions"

    request_body = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request_body.update(kwargs)

    logger.info(
        "Calling LiteLLM proxy at %s with requested_model=%s temperature=%s max_tokens=%s",
        api_url,
        model_name,
        temperature,
        max_tokens,
    )

    try:
        response = requests.post(
            api_url,
            json=request_body,
            headers=headers,
            timeout=60,
        )

        if response.status_code != 200:
            error_msg = response.text
            logger.error(
                "LiteLLM proxy returned error %d: %s",
                response.status_code,
                error_msg[:500],
            )
            raise RuntimeError(f"LiteLLM proxy error {response.status_code}: {error_msg}")

        response_data = response.json()

        if "choices" not in response_data or not response_data["choices"]:
            logger.error("No choices in response from proxy: %r", response_data)
            raise ValueError("No choices in response from proxy")

        choice = response_data["choices"][0]
        if "message" not in choice or "content" not in choice["message"]:
            logger.error("Unexpected response format from proxy: %r", response_data)
            raise ValueError("Unexpected response format from proxy")

        answer = choice["message"]["content"]
        if not answer:
            logger.error("Received empty response from proxy")
            raise ValueError("Received empty response from proxy")

        # Log actual model used for visibility / fallback tracking
        actual_model = response_data.get("model", "unknown")
        fallback_used = actual_model != model_name and actual_model != "unknown"

        if fallback_used:
            logger.warning(
                "LiteLLM fallback triggered | requested_model=%s | actual_model=%s",
                model_name,
                actual_model,
            )
        else:
            logger.info(
                "LiteLLM primary model used | requested_model=%s | actual_model=%s",
                model_name,
                actual_model,
            )

        logger.info("LiteLLM proxy response generated: %d chars", len(answer))
        logger.debug("LiteLLM response preview: %s", answer[:200])

        return answer

    except requests.exceptions.Timeout:
        logger.error("LiteLLM proxy request timed out after 60 seconds")
        raise RuntimeError("LiteLLM proxy request timed out")
    except requests.exceptions.ConnectionError as e:
        logger.error("Cannot connect to LiteLLM proxy at %s: %s", api_url, str(e))
        raise RuntimeError(f"Cannot connect to LiteLLM proxy at {api_url}")
    except Exception as e:
        logger.exception("LiteLLM proxy call failed for model=%s", model_name)
        raise RuntimeError(f"LiteLLM proxy request failed: {str(e)}") from e


def generate_answer(query: str, docs: List[Any]) -> str:
    """
    Generate an answer using RAG with LiteLLM proxy.
    """
    logger.info("Generating answer for query=%r using %d docs", query, len(docs))

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('filename', doc.metadata.get('source', 'unknown'))}\n{doc.page_content}"
            for doc in docs
        ]
    )

    prompt = f"""Answer using only the context below. If the answer cannot be found in the context, respond with exactly "I don't know".

Context:
{context}

Question:
{query}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval-augmented assistant. Answer using only the provided context. "
                "If the answer cannot be found in the context, respond with exactly 'I don't know'."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    return call_litellm(messages, temperature=0.2, max_tokens=1024)
