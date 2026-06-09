import os
import requests
from typing import List, Dict, Any

from rag.logger import logger


# ==============================
# ✅ LiteLLM HTTP call
# ==============================
def call_litellm(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.2,
    max_tokens: int = 1024
) -> str:
    proxy_url = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000").rstrip("/")
    api_url = f"{proxy_url}/v1/chat/completions"

    payload = {
        "model": model or os.getenv("LITELLM_MODEL", "nvidia-llm"),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info(
        "Calling LiteLLM proxy: %s | model=%s",
        api_url,
        payload["model"]
    )

    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('LITELLM_MASTER_KEY')}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=60
        )

        if response.status_code != 200:
            logger.error("LiteLLM error %s: %s", response.status_code, response.text)
            raise RuntimeError(response.text)

        data = response.json()

        if "choices" not in data or not data["choices"]:
            raise RuntimeError("Invalid response format from LiteLLM")

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        logger.exception("LiteLLM call failed")
        raise RuntimeError(f"LiteLLM error: {str(e)}")


# ==============================
# ✅ Role-based doc filtering
# ==============================
def _filter_docs_by_role(docs: List[Any], role: str) -> List[Any]:
    """
    Enforce document-level access control.
    """
    if role == "admin":
        return docs

    return [
        d for d in docs
        if not d.metadata.get("classified", False)
    ]


# ==============================
# ✅ RAG answer generation
# ==============================
def generate_answer(
    query: str,
    docs: List[Any],
    role: str,
    model: str = None
) -> str:
    """
    Generate answer using RAG + LiteLLM.
    The orchestrator can pass a LiteLLM model alias via `model`.
    """

    docs = _filter_docs_by_role(docs, role)

    if not docs:
        logger.info("No accessible docs found → returning I don't know")
        return "I don't know"

    logger.info(
        "Generating answer using %s docs with model=%s",
        len(docs),
        model or os.getenv("LITELLM_MODEL", "nvidia-llm")
    )

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('filename', doc.metadata.get('source', 'unknown'))}\n{doc.page_content}"
            for doc in docs
        ]
    )

    prompt = f"""Answer using only the context below.
If the answer cannot be found, respond with exactly: I don't know.

Context:
{context}

Question:
{query}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval-augmented assistant. "
                "Answer ONLY using the provided context. "
                "If the answer cannot be found in the context, return exactly: I don't know."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    return call_litellm(
        messages,
        model=model,
        temperature=0.2,
        max_tokens=1024
    )