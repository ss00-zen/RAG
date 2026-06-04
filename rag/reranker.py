import requests

API_URL = "https://integrate.api.nvidia.com/v1/rerank"
API_KEY = "YOUR_NVIDIA_API_KEY"


def rerank_docs(query, docs, top_k=5):

    if not docs:
        return docs

    # ✅ Prepare passages
    passages = [d.page_content for d in docs]

    payload = {
        "model": "nvidia/rerank-qa-mistral-4b",
        "query": query,
        "documents": passages
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, json=payload, headers=headers)

    if response.status_code != 200:
        print("Reranker error:", response.text)
        return docs  # fallback

    results = response.json()["data"]

    # ✅ Extract ranked indices
    ranked_indices = [r["index"] for r in results]

    # ✅ Reorder docs
    reranked_docs = [docs[i] for i in ranked_indices]

    return reranked_docs[:top_k]