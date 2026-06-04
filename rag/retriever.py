from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

def create_vector_store(chunks):
    embedding = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    db = FAISS.from_documents(chunks, embedding)
    return db

def retrieve_docs(db, query):
    retriever = db.as_retriever(search_kwargs={"k": 10})
    return retriever.invoke(query)