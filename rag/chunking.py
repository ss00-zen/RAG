from langchain_text_splitters import RecursiveCharacterTextSplitter


def hierarchical_chunking(docs, progress_callback=None):
    """
    Hierarchical chunking with controlled size and limits.
    Ensures:
    ✅ No chunk explosion
    ✅ Safe token size for NVIDIA embeddings
    ✅ Good semantic context
    """

    # ✅ Parent chunks (larger context)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # ~200–250 tokens
        chunk_overlap=150
    )

    # ✅ Child chunks (embedding-safe)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,    # ✅ critical (fits <512 token limit)
        chunk_overlap=80
    )

    parent_chunks = parent_splitter.split_documents(docs)

    final_chunks = []
    total = len(parent_chunks)

    for idx, parent in enumerate(parent_chunks):
        children = child_splitter.split_documents([parent])

        # ✅ HARD LIMIT → avoid explosion
        children = children[:2]

        for child in children:
            child.metadata = dict(parent.metadata)
            child.metadata["parent"] = parent.page_content[:200]
            final_chunks.append(child)

        if progress_callback and total:
            progress_callback((idx + 1) / total)

    if progress_callback and total == 0:
        progress_callback(1.0)

    return final_chunks