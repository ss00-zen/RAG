from rag.ingestion import load_pdf
from rag.chunking import hierarchical_chunking
from rag.hybrid import hybrid_search
from rag.query import rewrite_query, expand_query
from rag.reranker import rerank_docs


def trim_query(text, max_chars=1500):
    return text[:max_chars]


# ✅ DOCUMENT PROCESSING
def process_document(file_path, progress_callback=None):

    if progress_callback:
        progress_callback(0.05, "📄 Loading PDF document...")

    docs = load_pdf(file_path)

    if progress_callback:
        progress_callback(0.15, f"📄 Loaded {len(docs)} pages")

    def chunk_progress(fraction):
        if progress_callback:
            progress_callback(0.15 + 0.35 * fraction, "📄 Chunking document...")

    chunks = hierarchical_chunking(docs, progress_callback=chunk_progress)

    if progress_callback:
        progress_callback(0.50, f"✅ Chunks created: {len(chunks)}")

    return chunks


# ✅ QUERY PIPELINE (THIS IS MISSING RIGHT NOW)
def run_pipeline(query, db, bm25, chunks, messages, top_k=20, allow_classified=True):

    # ✅ Step 1: Build history context
    
    history = " ".join([
        m["content"][:200]   # limit each message length
        for m in messages[-3:]   # limit number of messages
        if m["role"] == "user"
    ])
    
    enhanced_query = history + " " + query
    enhanced_query = trim_query(enhanced_query)


    # ✅ Step 2: rewrite
    rewritten = rewrite_query(enhanced_query)

    # ✅ Step 3: expand
    expanded_queries = expand_query(rewritten)

    all_docs = []

    # ✅ Step 4: Retrieve using expanded queries
    for q in expanded_queries:
        q = trim_query(q)
        docs = hybrid_search(
            q,   # ✅ FIXED (use expanded query)
            db,
            bm25,
            chunks,
            allow_classified=allow_classified
        )

        all_docs.extend(docs)

    # ✅ Step 5: Deduplicate BEFORE rerank
    unique_docs = list({
        d.page_content: d for d in all_docs
    }.values())

    # ✅ Step 6: Rerank ONCE (global ranking)
    final_docs = rerank_docs(query, unique_docs, top_k=top_k)

    return final_docs