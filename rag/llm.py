"""
LiteLLM Gateway - HTTP Proxy Mode

This module makes HTTP requests to a LiteLLM proxy server instead of using the SDK directly.
The proxy is configured via litellm_config.yaml and runs on localhost:4000.

Prerequisites:
- LiteLLM proxy must be running: python start_litellm_proxy.ps1 (Windows) or bash start_litellm_proxy.sh (Unix)
- Endpoint: http://localhost:4000/v1/chat/completions

Environment variables:
- NVIDIA_API_KEY: NVIDIA API key (required, used by proxy)
- LITELLM_PROXY_URL: Proxy URL (default: http://localhost:4000)
- LITELLM_MODEL: Model name (default: nvidia-llm)
"""

import os
import json
import requests
from typing import Any, Dict, List, Optional

from rag.logger import logger


# Environment variable getter
def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if value else default.strip()


# Get proxy URL
def _get_proxy_url() -> str:
    """Get LiteLLM proxy base URL."""
    return _env("LITELLM_PROXY_URL", "http://localhost:4000")


# Check proxy health
def _check_proxy_health() -> bool:
    """Verify LiteLLM proxy is running and healthy."""
    proxy_url = _get_proxy_url()
    health_url = f"{proxy_url}/health"
    
    try:
        response = requests.get(health_url, timeout=5)
        is_healthy = response.status_code == 200
        
        if is_healthy:
            logger.debug("LiteLLM proxy is healthy at %s", proxy_url)
        else:
            logger.warning("LiteLLM proxy returned status %d at %s", response.status_code, health_url)
        
        return is_healthy
        
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to LiteLLM proxy at %s. Is it running?", proxy_url)
        logger.info("Start the proxy with: python start_litellm_proxy.ps1 (Windows) or bash start_litellm_proxy.sh (Unix)")
        return False
    except Exception as e:
        logger.error("Error checking proxy health at %s: %s", health_url, str(e))
        return False


# Main LLM completion function
def call_litellm(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    **kwargs,
) -> str:
    """
    Call LLM via LiteLLM HTTP proxy.
    
    The proxy handles routing to NVIDIA API based on litellm_config.yaml.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name (default: nvidia-llm, uses config from litellm_config.yaml)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        **kwargs: Additional arguments passed to proxy
    
    Returns:
        Response text from LLM
    
    Raises:
        EnvironmentError: If NVIDIA_API_KEY is not set
        RuntimeError: If proxy is unavailable or request fails
    """
    
    # Validate NVIDIA API key is set
    nvidia_api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not nvidia_api_key:
        raise EnvironmentError(
            "NVIDIA_API_KEY environment variable is not set or is empty. "
            "Please set a valid NVIDIA API key."
        )
    
    # Check proxy is running
    proxy_url = _get_proxy_url()
    if not _check_proxy_health():
        raise RuntimeError(
            f"LiteLLM proxy is not running at {proxy_url}. "
            "Please start it with: python start_litellm_proxy.ps1 (Windows) or bash start_litellm_proxy.sh (Unix)"
        )
    
    # Get model name
    model_name = model or _env("LITELLM_MODEL", "nvidia-llm")
    
    # Prepare request to proxy
    api_url = f"{proxy_url}/v1/chat/completions"
    
    request_body = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    # Add any additional kwargs (like top_p, frequency_penalty, etc.)
    request_body.update(kwargs)
    
    logger.info(
        "Calling LiteLLM proxy at %s with model=%s temperature=%s max_tokens=%s",
        api_url,
        model_name,
        temperature,
        max_tokens,
    )
    
    try:
        response = requests.post(
            api_url,
            json=request_body,
            timeout=60,
            headers={"Content-Type": "application/json"},
        )
        
        # Check for errors
        if response.status_code != 200:
            error_msg = response.text
            logger.error(
                "LiteLLM proxy returned error %d: %s",
                response.status_code,
                error_msg[:200],
            )
            raise RuntimeError(f"LiteLLM proxy error {response.status_code}: {error_msg}")
        
        # Parse response
        response_data = response.json()
        
        # Extract answer from OpenAI-compatible response format
        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                answer = choice["message"]["content"]
            else:
                logger.error("Unexpected response format from proxy: %r", response_data)
                raise ValueError("Unexpected response format from proxy")
        else:
            logger.error("No choices in response from proxy: %r", response_data)
            raise ValueError("No choices in response from proxy")
        
        if not answer:
            logger.error("Received empty response from proxy")
            raise ValueError("Received empty response from proxy")
        
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


# RAG answer generation
def generate_answer(query: str, docs: List[Any]) -> str:
    """
    Generate an answer using RAG with LiteLLM proxy.
    
    Args:
        query: User's question
        docs: List of retrieved document chunks
    
    Returns:
        Generated answer text
    """
    logger.info("Generating answer for query=%r using %d docs", query, len(docs))
    
    context = "\n\n".join([
        f"Source: {doc.metadata.get('filename', doc.metadata.get('source', 'unknown'))}\n{doc.page_content}"
        for doc in docs
    ])
    
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
