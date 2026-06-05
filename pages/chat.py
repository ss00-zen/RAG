import streamlit as st
import os
import pickle
import uuid
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from rag.logger import logger
from rag.retriever_dense import NvidiaEmbeddings
from rag.chat_store import init_db, save_message, load_messages, get_chat_titles
from rag.pipeline import run_pipeline
from rag.llm import generate_answer
from rag.eval_logger import log_interaction, load_logs
from rag.eval_dataset import build_dataset
from rag.evaluator import run_evaluation

# ✅ ==============================
# AUTH CHECK (MANDATORY)
# ==============================
user = st.session_state.get("user")

if not user:
    st.error("Please login")
    st.stop()

user_id = user.get("sub")

# ✅ ==============================
# INIT
# ==============================
load_dotenv()
init_db()
logger.info("Chat page initialized for user_id=%s", user_id)

roles = user.get("realm_access", {}).get("roles", [])
is_admin = "admin" in roles
logger.info("User roles=%s is_admin=%s", roles, is_admin)

# ✅ ==============================
# SESSION (USER ISOLATION ADDED)
# ==============================
if "session_id" not in st.session_state:
    st.session_state.session_id = f"{user_id}_{uuid.uuid4()}"
    logger.info("Created new session_id=%s", st.session_state.session_id)

if "messages" not in st.session_state:
    st.session_state.messages = load_messages(st.session_state.session_id)
    logger.info("Loaded messages for session_id=%s: %s", st.session_state.session_id, len(st.session_state.messages))

# ✅ ==============================
# SIDEBAR - CHAT THREADS
# ==============================
st.sidebar.title("💬 Chats")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    st.session_state.session_id = f"{user_id}_{uuid.uuid4()}"
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")

# ✅ ==============================
# EVALUATION
# ==============================
if st.button("📊 Run RAGAS Evaluation"):
    logs = load_logs(limit=3)
    dataset = build_dataset(logs)
    result = run_evaluation(dataset)

    st.subheader("📊 RAGAS Scores")
    st.json(result)

# ✅ ==============================
# LOAD SESSIONS
# ==============================
sessions = get_chat_titles(user_id)
logger.info("Loaded %s sessions for user_id=%s", len(sessions), user_id)

for session_id, title in sessions:
    title = (title[:35] + "...") if len(title) > 35 else title
    is_active = session_id == st.session_state.session_id

    if st.sidebar.button(
        title,
        key=session_id,
        use_container_width=True,
        type="primary" if is_active else "secondary"
    ):
        logger.info("Switching to session_id=%s for user_id=%s", session_id, user_id)
        st.session_state.session_id = session_id
        st.session_state.messages = load_messages(session_id)
        st.rerun()

# ✅ ==============================
# MAIN UI
# ==============================
st.title("💬 Chat with your Document")
st.markdown(f"**Active role:** {'admin' if is_admin else 'user'}")

# ✅ ==============================
# LOAD EMBEDDINGS
# ==============================
if "db" not in st.session_state:

    if os.path.exists("vectorstore"):

        with st.spinner("🔄 Loading embeddings..."):
            logger.info("Loading vectorstore and search indexes")

            embedding = NvidiaEmbeddings()

            db = FAISS.load_local(
                "vectorstore/",
                embedding,
                allow_dangerous_deserialization=True
            )

            with open("chunks.pkl", "rb") as f:
                chunks = pickle.load(f)
            logger.info("Loaded %s chunks from chunks.pkl", len(chunks))

            with open("bm25.pkl", "rb") as f:
                bm25 = pickle.load(f)
            logger.info("Loaded bm25 search index")

            st.session_state.db = db
            st.session_state.chunks = chunks
            st.session_state.bm25 = bm25

    else:
        logger.warning("Vectorstore missing; no document found")
        st.warning("⚠️ No document found. Please upload first.")
        st.stop()

# ✅ ==============================
# CHAT HISTORY
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ ==============================
# CLASSIFICATION CONTROL
# ==============================
if is_admin:
    allow_classified = st.checkbox(
        "🔐 Include Classified Documents",
        value=True
    )
else:
    allow_classified = False
    st.info("🔒 Classified documents are only available to admin users.")

# ✅ ==============================
# CHAT INPUT
# ==============================
query = st.chat_input("Ask your question")

if query:

    st.session_state.messages.append({"role": "user", "content": query})
    save_message(st.session_state.session_id, "user", query)

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("🧠 Thinking..."):
        logger.info("Running pipeline for user_id=%s query=%r allow_classified=%s is_admin=%s", user_id, query, allow_classified, is_admin)

        docs = run_pipeline(
            query,
            st.session_state.db,
            st.session_state.bm25,
            st.session_state.chunks,
            st.session_state.messages,
            allow_classified=allow_classified,
            can_view_classified=is_admin,
            user_id=user_id   # ✅ critical: user isolation
        )
        logger.info("Pipeline returned %s documents", len(docs))

        answer = generate_answer(query, docs)
        logger.info("Answer generated, length=%s", len(answer) if answer else 0)
        log_interaction(query, answer, docs)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_message(st.session_state.session_id, "assistant", answer)

    with st.chat_message("assistant"):
        st.markdown(answer)

    with st.expander("📌 Retrieved Context"):
        for doc in docs:
            source = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")
            st.markdown(f"**Source:** {source}")
            st.write(doc.page_content)
            st.markdown("---")
        st.write(len(docs), "documents retrieved")