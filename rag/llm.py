import os
import requests

def generate_answer(query, docs):

    api_key = os.getenv("NVIDIA_API_KEY")

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

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

    if response.status_code != 200:
        return f"Error: {response.text}"

    result = response.json()
    return result["choices"][0]["message"]["content"]