import streamlit as st
import os
import pickle
import uuid
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from rag.retriever_dense import NvidiaEmbeddings
from rag.chat_store import init_db, save_message, load_messages, get_chat_titles
from rag.pipeline import run_pipeline
from rag.llm import generate_answer
from rag.chat_store import get_chat_titles, load_messages
from rag.eval_logger import log_interaction

from rag.eval_logger import load_logs
from rag.eval_dataset import build_dataset
from rag.evaluator import run_evaluation

import uuid

load_dotenv()
init_db()

# ✅ Session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = load_messages(st.session_state.session_id)

# ==============================
# ✅ SIDEBAR - CHAT THREADS
# ==============================
st.sidebar.title("💬 Chats")

# ✅ New chat button
if st.sidebar.button("➕ New Chat", use_container_width=True):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")



if st.button("📊 Run RAGAS Evaluation"):

    logs = load_logs(limit=3)   # ✅ last 50 interactions

    dataset = build_dataset(logs)

    result = run_evaluation(dataset)

    st.subheader("📊 RAGAS Scores")
    st.json(result)



# ✅ Load sessions
sessions = get_chat_titles()

# ✅ Display threads
for session_id, title in sessions:

    # truncate
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


st.title("💬 Chat with your Document")






# ✅ Load embeddings
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

    else:
        st.warning("⚠️ No document found. Please upload first.")
        st.stop()

# ✅ Chat history display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ Classification control in main pane
allow_classified = st.checkbox(
    "🔐 Include Classified Documents",
    value=True
)

# ✅ Input
query = st.chat_input("Ask your question")

if query:

    st.session_state.messages.append({"role": "user", "content": query})
    save_message(st.session_state.session_id, "user", query)

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("🧠 Thinking..."):

        docs = run_pipeline(
            query,
            st.session_state.db,
            st.session_state.bm25,
            st.session_state.chunks,
            st.session_state.messages,
            allow_classified=allow_classified
        )

        answer = generate_answer(query, docs)
        log_interaction(query, answer, docs)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_message(st.session_state.session_id, "assistant", answer)

    with st.chat_message("assistant"):
        st.markdown(answer)

    with st.expander("📌 Retrieved Context"):
        for doc in docs:
            st.write(doc.page_content)
        st.write(len(docs), "documents retrieved")