import os

from rag.ingestion import load_pdf
from rag.chunking import hierarchical_chunking
from rag.hybrid import hybrid_search
from rag.logger import logger
from rag.query import rewrite_query, expand_query
from rag.reranker import rerank_docs


def trim_query(text, max_chars=1500):
    return text[:max_chars]


# ✅ DOCUMENT PROCESSING
def process_document(file_path, source_name=None, progress_callback=None):

    if progress_callback:
        progress_callback(0.05, "📄 Loading PDF document...")

    source_name = source_name or os.path.basename(file_path)
    logger.info("Loading document %s with source_name=%s", file_path, source_name)
    docs = load_pdf(file_path, source_name=source_name)

    if progress_callback:
        progress_callback(0.15, f"📄 Loaded {len(docs)} pages")
    logger.debug("Loaded %s page documents", len(docs))

    def chunk_progress(fraction):
        if progress_callback:
            progress_callback(0.15 + 0.35 * fraction, "📄 Chunking document...")

    chunks = hierarchical_chunking(docs, progress_callback=chunk_progress)

    for chunk in chunks:
        chunk.metadata["filename"] = source_name

    logger.info("Created %s chunks for source %s", len(chunks), source_name)
    if progress_callback:
        progress_callback(0.50, f"✅ Chunks created: {len(chunks)}")

    return chunks


# ✅ QUERY PIPELINE (THIS IS MISSING RIGHT NOW)
def run_pipeline(query, db, bm25, chunks, messages, top_k=20, allow_classified=True, can_view_classified=True, user_id=None, debug_info=None):

    # honor explicit view permission
    if not can_view_classified:
        allow_classified = False

    # ✅ Step 1: Build history context
    
    history = " ".join([
        m["content"][:200]   # limit each message length
        for m in messages[-3:]   # limit number of messages
        if m["role"] == "user"
    ])
    logger.info("Running query pipeline for user_id=%s query=%r", user_id, query)
    logger.debug("History context length=%s user messages=%s", len(history), len([m for m in messages if m["role"] == "user"]))
    
    enhanced_query = history + " " + query
    enhanced_query = trim_query(enhanced_query)

    # ✅ Step 2: use the original query for retrieval
    expanded_queries = [enhanced_query]
    logger.info("Using original query for retrieval; rewrites/expansions disabled")

    all_docs = []
    logger.info("Retrieval queries count=%s", len(expanded_queries))

    # ✅ Step 4: Retrieve using the original query
    for q in expanded_queries:
        q = trim_query(q)
        logger.debug("Retrieving docs for query=%r", q)
        docs = hybrid_search(
            q,
            db,
            bm25,
            chunks,
            allow_classified=allow_classified,
            debug_info=debug_info,
        )
        logger.info("Retrieved %s docs for query", len(docs))

        all_docs.extend(docs)

    # ✅ Step 5: Deduplicate BEFORE rerank
    unique_docs = list({
        d.page_content: d for d in all_docs
    }.values())

    if not allow_classified:
        filtered_docs = [
            d for d in unique_docs
            if not d.metadata.get("classified", False)
        ]
        logger.info("Filtered classified docs: %s -> %s", len(unique_docs), len(filtered_docs))
        if debug_info is not None:
            debug_info.append(f"final_docs filtered classified: {len(unique_docs)} -> {len(filtered_docs)}")
        unique_docs = filtered_docs

    # ✅ Step 6: Rerank ONCE (global ranking)
    final_docs = rerank_docs(query, unique_docs, top_k=top_k)

    return final_docs