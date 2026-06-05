import os
import requests
from rag.logger import logger

def generate_answer(query, docs):
    logger.info("Generating answer for query=%r using %s docs", query, len(docs))

    api_key = os.getenv("NVIDIA_API_KEY")

    context = "\n\n".join([
        f"Source: {doc.metadata.get('filename', doc.metadata.get('source', 'unknown'))}\n{doc.page_content}"
        for doc in docs
    ])

    prompt = f"""
Answer using only the context below. If the answer cannot be found in the context, respond with exactly "I don't know".

Context:
{context}

Question:
{query}
"""

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    response = requests.post(url, headers=headers, json=data)
    logger.info("LLM request sent, status=%s", response.status_code)

    if response.status_code != 200:
        logger.error("LLM API error: %s", response.text)
        return f"Error: {response.text}"

    result = response.json()
    answer = result["choices"][0]["message"]["content"]
    logger.info("LLM answer generated length=%s", len(answer))
    logger.debug("LLM answer content: %r", answer[:200])
    return answer