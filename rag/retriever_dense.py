import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from rag.logger import logger


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

        logger.debug("Requesting document embeddings for %s texts", len(texts))
        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            logger.error("Embedding error: %s", response.text)
            raise Exception(f"Embedding error: {response.text}")

        result = response.json()
        logger.debug("Received document embeddings response")

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

        logger.debug("Requesting query embedding")
        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            logger.error("Query embedding error: %s", response.text)
            raise Exception(f"Query embedding error: {response.text}")

        result = response.json()
        logger.debug("Received query embedding response")

        return result["data"][0]["embedding"]


def build_vector_db(chunks, progress_callback=None):
    logger.info("Building vector DB for %s chunks", len(chunks))
    embedding = NvidiaEmbeddings()

    # ✅ FIX: embed documents correctly
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    db = FAISS.from_texts(
        texts,
        embedding,
        metadatas=metadatas
    )

    logger.info("Vector DB build complete")
    return db



def dense_search(db, query, k=20):
    logger.info("Performing dense search for query=%r k=%s", query, k)
    retriever = db.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(query)
    logger.info("Dense search returned %s hits", len(results))
    return results
