from typing import TypedDict, List, Dict, Optional

from langgraph.graph import StateGraph, END

from rag.logger import logger


# IMPORTANT:
# replace these imports with your actual init/load functions
# Example:
# from rag.store import get_db, get_bm25, get_chunks

# ---------------------------------------------------
# GLOBALS / SINGLETONS
# Load these once at startup in real app
# ---------------------------------------------------
db = None
bm25 = None
chunks = None


def set_runtime_objects(_db, _bm25, _chunks):
    global db, bm25, chunks
    db = _db
    bm25 = _bm25
    chunks = _chunks


class GraphState(TypedDict, total=False):
    query: str
    role: str
    user_id: Optional[str]
    messages: List[Dict[str, str]]

    classification: str
    allowed: bool
    allow_classified: bool

    docs: list
    answer: str
    debug_info: List[str]


# -----------------------------
# NODE 1: classify query
# -----------------------------
def classify_query_node(state: GraphState) -> GraphState:
    query = state["query"].lower()

    sensitive_keywords = [
        "salary",
        "confidential",
        "internal",
        "pii",
        "client data",
        "private",
        "restricted",
        "financial statement",
        "customer record",
    ]

    classification = "classified" if any(k in query for k in sensitive_keywords) else "non_classified"

    debug_info = state.get("debug_info", [])
    debug_info.append(f"classification={classification}")

    logger.info("Classified query=%r as %s", state["query"], classification)

    return {
        "classification": classification,
        "debug_info": debug_info,
    }


# -----------------------------
# NODE 2: access check
# -----------------------------
def access_check_node(state: GraphState) -> GraphState:
    role = state["role"]
    classification = state["classification"]

    allowed = not (classification == "classified" and role != "admin")
    allow_classified = role == "admin"

    debug_info = state.get("debug_info", [])
    debug_info.append(f"role={role}")
    debug_info.append(f"allowed={allowed}")
    debug_info.append(f"allow_classified={allow_classified}")

    logger.info(
        "Access check role=%s classification=%s allowed=%s",
        role, classification, allowed
    )

    return {
        "allowed": allowed,
        "allow_classified": allow_classified,
        "debug_info": debug_info,
    }


def access_router(state: GraphState) -> str:
    return "retrieve" if state["allowed"] else "deny"


# -----------------------------
# NODE 3A: deny
# -----------------------------
def deny_node(state: GraphState) -> GraphState:
    logger.warning("Access denied for user_id=%s query=%r", state.get("user_id"), state["query"])
    return {
        "answer": "❌ You are not authorized to access this information."
    }


# -----------------------------
# NODE 3B: retrieve
# -----------------------------
def retrieve_node(state: GraphState) -> GraphState:
    # ✅ IMPORT INSIDE FUNCTION (fix circular import)
    from rag.pipeline import run_pipeline

    if db is None or bm25 is None or chunks is None:
        raise RuntimeError("Runtime objects not initialized")

    debug_info = state.get("debug_info", [])

    docs = run_pipeline(
        query=state["query"],
        db=db,
        bm25=bm25,
        chunks=chunks,
        messages=state.get("messages", []),
        role=state["role"],
        allow_classified=state["allow_classified"],
        can_view_classified=state["allow_classified"],
        user_id=state.get("user_id"),
        debug_info=debug_info,
    )

    return {
        "docs": docs,
        "debug_info": debug_info,
    }


# -----------------------------
# NODE 4: generate answer
# -----------------------------
def generate_node(state: GraphState) -> GraphState:
    # ✅ IMPORT INSIDE FUNCTION
    from rag.llm import generate_answer

    docs = state.get("docs", [])

    if not docs:
        return {"answer": "I don't know"}

    answer = generate_answer(
        query=state["query"],
        docs=docs,
        role=state["role"],
    )

    return {"answer": answer}


# -----------------------------
# BUILD GRAPH
# -----------------------------
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify", classify_query_node)
    graph.add_node("access_check", access_check_node)
    graph.add_node("deny", deny_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "access_check")

    graph.add_conditional_edges(
        "access_check",
        access_router,
        {
            "deny": "deny",
            "retrieve": "retrieve",
        }
    )

    graph.add_edge("deny", END)
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


app_graph = build_graph()


def orchestrate(query: str, role: str, user_id: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None):
    result = app_graph.invoke({
        "query": query,
        "role": role,
        "user_id": user_id,
        "messages": messages or [],
        "debug_info": [],
    })

    return {
    "answer": result.get("answer"),
    "docs": result.get("docs", []),
    "debug_info": result.get("debug_info", []),
    "classification": result.get("classification"),
    "allowed": result.get("allowed")}
