from rank_bm25 import BM25Okapi

def build_bm25(chunks):
    corpus = [doc.page_content.split() for doc in chunks]
    return BM25Okapi(corpus)

def bm25_search(bm25, chunks, query):
    scores = bm25.get_scores(query.split())

    ranked = sorted(
        list(zip(chunks, scores)),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:10]]