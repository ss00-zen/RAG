import streamlit as st
import os
import pickle
import uuid
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from rag.logger import logger
from rag.retriever_dense import NvidiaEmbeddings
from rag.chat_store import init_db, save_message, load_messages, get_chat_titles
from rag.orchestrator import orchestrate, set_runtime_objects  # ✅ important
from rag.eval_logger import log_interaction, load_logs
from rag.eval_dataset import build_dataset
from rag.evaluator import run_evaluation


# ==============================
# ✅ AUTH CHECK
# ==============================
user = st.session_state.get("user")

if not user:
    st.error("Please login")
    st.stop()

user_id = user.get("sub")


# ==============================
# ✅ INIT
# ==============================
load_dotenv(override=True)
init_db()

roles = user.get("realm_access", {}).get("roles", [])
is_admin = "admin" in roles

logger.info("Chat initialized | user=%s role=%s", user_id, roles)


# ==============================
# ✅ SESSION
# ==============================
if "session_id" not in st.session_state:
    st.session_state.session_id = f"{user_id}_{uuid.uuid4()}"

if "messages" not in st.session_state:
    st.session_state.messages = load_messages(st.session_state.session_id)


# ==============================
# ✅ SIDEBAR
# ==============================
st.sidebar.title("💬 Chats")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    st.session_state.session_id = f"{user_id}_{uuid.uuid4()}"
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")


# ==============================
# ✅ EVALUATION
# ==============================
if st.button("📊 Run RAGAS Evaluation"):
    logs = load_logs(limit=3)
    dataset = build_dataset(logs)
    result = run_evaluation(dataset)

    st.subheader("📊 RAGAS Scores")
    st.json(result)


# ==============================
# ✅ LOAD CHAT SESSIONS
# ==============================
sessions = get_chat_titles(user_id)

for session_id, title in sessions:
    title = (title[:35] + "...") if len(title) > 35 else title
    is_active = session_id == st.session_state.session_id

    if st.sidebar.button(
        title,
        key=session_id,
        use_container_width=True,
        type="primary" if is_active else "secondary"
    ):
        st.session_state.session_id = session_id
        st.session_state.messages = load_messages(session_id)
        st.rerun()


# ==============================
# ✅ MAIN UI
# ==============================
st.title("💬 Chat with your Document")
st.markdown(f"**Active role:** {'admin' if is_admin else 'user'}")


# ==============================
# ✅ LOAD VECTOR DB + SET ORCHESTRATOR RUNTIME
# ==============================
if "db" not in st.session_state:

    if os.path.exists("vectorstore"):

        with st.spinner("🔄 Loading embeddings..."):

            embedding = NvidiaEmbeddings()

            db = FAISS.load_local(
                "vectorstore/",
                embedding,
                allow_dangerous_deserialization=True
            )

            with open("chunks.pkl", "rb") as f:
                chunks = pickle.load(f)

            with open("bm25.pkl", "rb") as f:
                bm25 = pickle.load(f)

            st.session_state.db = db
            st.session_state.chunks = chunks
            st.session_state.bm25 = bm25

            # ✅ ✅ CRITICAL FIX
            set_runtime_objects(db, bm25, chunks)

    else:
        st.warning("⚠️ No document found. Please upload first.")
        st.stop()


# ==============================
# ✅ DISPLAY HISTORY
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ==============================
# ✅ INPUT
# ==============================
query = st.chat_input("Ask your question")

if query:

    # save user message
    st.session_state.messages.append({"role": "user", "content": query})
    save_message(st.session_state.session_id, "user", query)

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("🧠 Thinking..."):

        logger.info(
            "Calling orchestrator | user=%s role=%s query=%s",
            user_id,
            "admin" if is_admin else "user",
            query
        )

        result = orchestrate(
            query=query,
            role="admin" if is_admin else "user",
            user_id=user_id,
            messages=st.session_state.messages
        )

        answer = result.get("answer", "No answer returned")
        docs = result.get("docs", [])

        log_interaction(query, answer, docs)

    # save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_message(st.session_state.session_id, "assistant", answer)

    with st.chat_message("assistant"):
        st.markdown(answer)

    # ==============================
    # ✅ CONTEXT VIEW
    # ==============================
    with st.expander("📌 Retrieved Context"):
        if docs:
            for doc in docs:
                source = doc.metadata.get("filename") or doc.metadata.get("source", "unknown")
                st.markdown(f"**Source:** {source}")
                st.write(doc.page_content)
                st.markdown("---")
            st.write(len(docs), "documents retrieved")
        else:
            st.write("No documents retrieved")