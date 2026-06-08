import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from rag.logger import logger
from rag.retriever_dense import NvidiaEmbeddings, build_vector_db
from rag.pipeline import process_document
from rag.retriever_sparse import build_bm25

# ==============================
# ✅ AUTH CHECK (ADMIN ONLY)
# ==============================
user = st.session_state.get("user")

if not user:
    st.error("Please login")
    st.stop()

roles = user.get("realm_access", {}).get("roles", [])
is_admin = "admin" in roles

if not is_admin:
    st.error("🚫 Access denied. Admin only.")
    st.stop()

# ==============================
# ✅ INIT
# ==============================
load_dotenv(override=True)

st.title("📄 Upload & Process Document")

# ==============================
# ✅ FORM
# ==============================
with st.form("upload_form"):

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    # ✅ classification toggle
    is_classified = st.checkbox("🔐 Mark as Classified", value=False)

    submitted = st.form_submit_button(
        "🚀 Process Document",
        use_container_width=True
    )

# ==============================
# ✅ VALIDATION
# ==============================
if submitted and uploaded_file is None:
    st.warning("⚠️ Please upload a file before submitting")

# ==============================
# ✅ PROCESS
# ==============================
if submitted and uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    logger.info("Uploaded file %s saved to temp.pdf", uploaded_file.name)

    with st.status("Processing document...", expanded=True) as status:
        try:
            # STEP 1: Chunking
            status.write("📄 Chunking document...")
            logger.info("Starting document processing for %s", uploaded_file.name)
            new_chunks = process_document("temp.pdf", source_name=uploaded_file.name)
            logger.info("Document processed into %s chunks", len(new_chunks))

            # ✅ attach classification metadata
            for chunk in new_chunks:
                chunk.metadata["classified"] = is_classified

            status.write(f"✅ {len(new_chunks)} chunks created")

            # ✅ Load existing chunks and merge
            import os
            all_chunks = []
            if os.path.exists("chunks.pkl"):
                with open("chunks.pkl", "rb") as f:
                    all_chunks = pickle.load(f)
                logger.info("Loaded %s existing chunks from chunks.pkl", len(all_chunks))
            
            all_chunks.extend(new_chunks)
            logger.info("Merged: now have %s total chunks", len(all_chunks))

            # STEP 2: Embeddings
            status.write("🧠 Generating embeddings...")
            logger.info("Building vector database for %s chunks", len(all_chunks))
            db = build_vector_db(all_chunks)
            logger.info("Vector database built")

            # STEP 3: BM25
            status.write("⚡ Building search index...")
            logger.info("Building BM25 index for %s chunks", len(all_chunks))
            bm25 = build_bm25(all_chunks)
            logger.info("BM25 index built")

            # STEP 4: Save
            status.write("💾 Saving...")
            logger.info("Saving %s merged chunks and index to disk", len(all_chunks))

            db.save_local("vectorstore/")

            with open("chunks.pkl", "wb") as f:
                pickle.dump(all_chunks, f)

            with open("bm25.pkl", "wb") as f:
                pickle.dump(bm25, f)

            status.update(label="✅ Done", state="complete")
            st.success("✅ Document processed! Go to Chat page.")
        except Exception as e:
            logger.exception("Document processing failed for %s", uploaded_file.name)
            st.error(f"🚫 Processing failed: {e}")
