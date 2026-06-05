from rag.logger import logger
from rag.retriever_dense import dense_search
from rag.retriever_sparse import bm25_search


def hybrid_search(query, db, bm25, chunks, allow_classified=True, debug_info=None):

    logger.info("Starting hybrid search for query=%r allow_classified=%s", query, allow_classified)
    dense_docs_orig = dense_search(db, query)
    sparse_docs_orig = bm25_search(bm25, chunks, query)
    logger.info("Dense search returned %s docs, sparse search returned %s docs", len(dense_docs_orig), len(sparse_docs_orig))

    dense_docs = dense_docs_orig
    sparse_docs = sparse_docs_orig

    # Treat missing 'classified' metadata as False (non-classified)
    if not allow_classified:
        dense_docs = [
            d for d in dense_docs_orig
            if not d.metadata.get("classified", False)
        ]

        sparse_docs = [
            d for d in sparse_docs_orig
            if not d.metadata.get("classified", False)
        ]

    if debug_info is not None:
        debug_info.append(f"hybrid_search(query={query!r}, allow_classified={allow_classified})")
        debug_info.append(f"  dense_docs: {len(dense_docs_orig)} -> {len(dense_docs)}")
        debug_info.append(f"  sparse_docs: {len(sparse_docs_orig)} -> {len(sparse_docs)}")
        if not allow_classified:
            filtered_dense = [
                d.page_content[:120].replace("\n", " ")
                for d in dense_docs_orig
                if d.metadata.get("classified", False)
            ]
            filtered_sparse = [
                d.page_content[:120].replace("\n", " ")
                for d in sparse_docs_orig
                if d.metadata.get("classified", False)
            ]
            if filtered_dense:
                debug_info.append(f"  filtered classified dense docs ({len(filtered_dense)}):")
                debug_info.extend([f"    {i+1}. {txt}" for i, txt in enumerate(filtered_dense[:3])])
            if filtered_sparse:
                debug_info.append(f"  filtered classified sparse docs ({len(filtered_sparse)}):")
                debug_info.extend([f"    {i+1}. {txt}" for i, txt in enumerate(filtered_sparse[:3])])

    # ✅ Score fusion (real)
    logger.debug("Combining dense and sparse scores")
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

    logger.info("Hybrid search returning %s docs", len(final_docs[:10]))
    return final_docs[:10]