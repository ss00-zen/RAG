from rag.retriever_dense import dense_search
from rag.retriever_sparse import bm25_search

def hybrid_search(query, db, bm25, chunks, allow_classified=True):

    dense_docs = dense_search(db, query)
    sparse_docs = bm25_search(bm25, chunks, query)

    
    if not allow_classified:
        dense_docs = [
            d for d in dense_docs
            if not d.metadata.get("classified", False)
        ]

        sparse_docs = [
            d for d in sparse_docs
            if not d.metadata.get("classified", False)
        ]


    # ✅ Score fusion (real)
    combined = {}

    for i, doc in enumerate(dense_docs):
        combined[doc.page_content] = 1 / (i + 1)

    for i, doc in enumerate(sparse_docs):
        combined[doc.page_content] = combined.get(doc.page_content, 0) + 1 / (i + 1)

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    final_docs = []
    seen = set()

    for text, _ in ranked:
        for d in dense_docs + sparse_docs:
            if d.page_content == text and text not in seen:
                final_docs.append(d)
                seen.add(text)

    return final_docs[:10]