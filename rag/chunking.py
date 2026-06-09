from langchain_text_splitters import RecursiveCharacterTextSplitter


def hierarchical_chunking(docs, progress_callback=None):
    """
    Hierarchical chunking with controlled size and limits.

    Ensures:
    ✅ No chunk explosion
    ✅ Safe token size for embeddings
    ✅ Metadata consistency (classified always present)
    """

    # ✅ Parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    # ✅ Child chunks
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )

    parent_chunks = parent_splitter.split_documents(docs)

    final_chunks = []
    total = len(parent_chunks)

    for idx, parent in enumerate(parent_chunks):
        children = child_splitter.split_documents([parent])

        # ✅ prevent explosion
        children = children[:2]

        for child in children:

            # ✅ copy metadata safely
            parent_metadata = dict(parent.metadata) if parent.metadata else {}

            # ✅ enforce default classification rule
            # missing → False (non-classified)
            parent_metadata["classified"] = parent_metadata.get("classified", False)

            # ✅ assign metadata
            child.metadata = parent_metadata

            # ✅ add parent context
            child.metadata["parent"] = parent.page_content[:200]

            final_chunks.append(child)

        if progress_callback and total:
            progress_callback((idx + 1) / total)

    if progress_callback and total == 0:
        progress_callback(1.0)

    return final_chunks
