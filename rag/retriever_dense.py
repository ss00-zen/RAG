import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


class NvidiaEmbeddings(Embeddings):

    def embed_documents(self, texts):

        url = "https://integrate.api.nvidia.com/v1/embeddings"

        headers = {
            "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "input": texts,
            "input_type": "passage"   # ✅ IMPORTANT (correct for docs)
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            raise Exception(f"Embedding error: {response.text}")

        result = response.json()

        return [item["embedding"] for item in result["data"]]


    def embed_query(self, text):

        url = "https://integrate.api.nvidia.com/v1/embeddings"

        headers = {
            "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "input": [text],
            "input_type": "query"   # ✅ IMPORTANT
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            raise Exception(f"Query embedding error: {response.text}")

        result = response.json()

        return result["data"][0]["embedding"]


def build_vector_db(chunks, progress_callback=None):
    embedding = NvidiaEmbeddings()

    # ✅ FIX: embed documents correctly
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    db = FAISS.from_texts(
        texts,
        embedding,
        metadatas=metadatas
    )

    return db



def dense_search(db, query, k=20):
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def build_vector_db(chunks):
    embedding = NvidiaEmbeddings()
    return FAISS.from_documents(chunks, embedding)
