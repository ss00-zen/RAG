import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from rag.retriever_dense import NvidiaEmbeddings, build_vector_db
from rag.pipeline import process_document
from rag.retriever_sparse import build_bm25

load_dotenv()

st.title("📄 Upload & Process Document")

# ✅ FORM START
with st.form("upload_form"):

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    # ✅ classification toggle
    is_classified = st.checkbox("🔐 Mark as Classified", value=False)

    # ✅ submit button
    submitted = st.form_submit_button(
        "🚀 Process Document",
        use_container_width=True
    )

# ✅ Validation
if submitted and uploaded_file is None:
    st.warning("⚠️ Please upload a file before submitting")

# ✅ PROCESS ONLY AFTER SUBMIT
if submitted and uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.status("Processing document...", expanded=True) as status:

        # STEP 1: Chunking
        status.write("📄 Chunking document...")
        chunks = process_document("temp.pdf")

        # ✅ attach classification metadata
        for chunk in chunks:
            chunk.metadata["classified"] = is_classified

        status.write(f"✅ {len(chunks)} chunks created")

        # STEP 2: Embeddings
        status.write("🧠 Generating embeddings...")
        db = build_vector_db(chunks)

        # STEP 3: BM25
        status.write("⚡ Building search index...")
        bm25 = build_bm25(chunks)

        # STEP 4: Save
        status.write("💾 Saving...")

        db.save_local("vectorstore/")

        with open("chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

        with open("bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)

        status.update(label="✅ Done", state="complete")

    st.success("✅ Document processed! Go to Chat page.")