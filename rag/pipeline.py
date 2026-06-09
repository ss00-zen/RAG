import os

from rag.ingestion import load_pdf
from rag.chunking import hierarchical_chunking
from rag.hybrid import hybrid_search
from rag.logger import logger
from rag.reranker import rerank_docs


# ==============================
# ✅ helper
# ==============================
def trim_query(text, max_chars=1500):
    return text[:max_chars]


# ==============================
# ✅ DOCUMENT PROCESSING
# ==============================
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

        # ✅ Safety: missing metadata → non-classified
        if "classified" not in chunk.metadata:
            chunk.metadata["classified"] = False

    logger.info("Created %s chunks for source %s", len(chunks), source_name)

    if progress_callback:
        progress_callback(0.50, f"✅ Chunks created: {len(chunks)}")

    return chunks


# ==============================
# ✅ QUERY PIPELINE (RETRIEVAL ONLY)
# ==============================
def run_pipeline(
    query,
    db,
    bm25,
    chunks,
    messages,
    role,   # ✅ required
    top_k=20,
    allow_classified=True,
    can_view_classified=True,
    user_id=None,
    debug_info=None
):

    # ✅ safety fallback (avoid crash if None passed)
    messages = messages or []

    # ✅ final permission flag
    if not can_view_classified:
        allow_classified = False

    logger.info(
        "Pipeline start | user_id=%s role=%s allow_classified=%s query=%r",
        user_id,
        role,
        allow_classified,
        query,
    )

    # ==============================
    # ✅ STEP 1: Build context from history
    # ==============================
    history = " ".join([
        m.get("content", "")[:200]
        for m in messages[-3:]
        if m.get("role") == "user"
    ])

    logger.debug("History length=%s", len(history))

    enhanced_query = trim_query(history + " " + query)

    # ==============================
    # ✅ STEP 2: Use only original query (no rewrite yet)
    # ==============================
    queries = [enhanced_query]

    all_docs = []

    # ==============================
    # ✅ STEP 3: Hybrid retrieval
    # ==============================
    for q in queries:
        q = trim_query(q)

        docs = hybrid_search(
            q,
            db,
            bm25,
            chunks,
            allow_classified=allow_classified,
            debug_info=debug_info,
        )

        logger.info("Retrieved %s docs", len(docs))

        all_docs.extend(docs)

    # ==============================
    # ✅ STEP 4: Deduplicate
    # ==============================
    unique_docs = list({
        d.page_content: d for d in all_docs
    }.values())

    # ==============================
    # ✅ STEP 5: Defensive filtering (double safety)
    # ==============================
    if not allow_classified:
        before = len(unique_docs)

        unique_docs = [
            d for d in unique_docs
            if not d.metadata.get("classified", False)
        ]

        after = len(unique_docs)

        logger.info("Filtered classified docs: %s -> %s", before, after)

        if debug_info is not None:
            debug_info.append(f"filtered: {before} -> {after}")

    # ==============================
    # ✅ STEP 6: Rerank
    # ==============================
    final_docs = rerank_docs(query, unique_docs, top_k=top_k)

    logger.info("Pipeline returning %s docs", len(final_docs))

    return final_docs