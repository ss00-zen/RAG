import os
import requests


def _call_nvidia(prompt):

    api_key = os.getenv("NVIDIA_API_KEY")

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
        "max_tokens": 512
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# ✅ Rewrite
def rewrite_query(query):
    prompt = f"""
Rewrite this query to improve retrieval:

Query: {query}
"""
    return _call_nvidia(prompt)


# ✅ Expand
def expand_query(query):
    prompt = f"""
Generate 3 alternative search queries:

Original query: {query}

Return each on a new line.
"""
    response = _call_nvidia(prompt)

    expanded = [q.strip() for q in response.split("\n") if q.strip()]
    return [query] + expanded
