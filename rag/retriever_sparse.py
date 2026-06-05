from rank_bm25 import BM25Okapi
from rag.logger import logger

def build_bm25(chunks):
    logger.info("Building BM25 index for %s chunks", len(chunks))
    corpus = [doc.page_content.split() for doc in chunks]
    bm25 = BM25Okapi(corpus)
    logger.info("BM25 index build complete")
    return bm25

def bm25_search(bm25, chunks, query):
    logger.info("Performing BM25 search for query=%r", query)
    scores = bm25.get_scores(query.split())

    ranked = sorted(
        list(zip(chunks, scores)),
        key=lambda x: x[1],
        reverse=True
    )
    logger.info("BM25 search returned %s hits", len(ranked))

    return [doc for doc, _ in ranked[:10]]