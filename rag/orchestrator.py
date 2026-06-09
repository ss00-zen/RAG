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

MODEL_POLICY = {
    "low": {
        "primary": "nvidia-fast",
        "fallback": "nvidia-balanced"
    },
    "medium": {
        "primary": "nvidia-balanced",
        "fallback": "nvidia-strong"
    },
    "high": {
        "primary": "nvidia-strong",
        "fallback": "nvidia-balanced"
    }
}


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

    complexity: str
    model_alias: str
    fallback_model_alias: str

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
        "AWS"
    ]

    
    import re

    clean_query = re.sub(r"[^\w\s]", "", query)

    classification = "classified" if any(
        k.lower() in clean_query for k in sensitive_keywords
    ) else "non_classified"


    debug_info = state.get("debug_info", [])
    debug_info.append(f"classification={classification}")

    logger.info("Classified query=%r as %s", state["query"], classification)
    
    logger.info("clean_query=%s", clean_query)
    logger.info("matched_keywords=%s", [
        k for k in sensitive_keywords if k.lower() in clean_query
    ])


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
# NODE 4: Model complexity Node
# -----------------------------
def classify_complexity_node(state: GraphState) -> GraphState:
    query = state["query"].strip().lower()

    
    import re

    query = state["query"].strip().lower()

    # normalize spaces + remove punctuation
    clean_query = re.sub(r"[^\w\s]", " ", query)
    query = re.sub(r"\s+", " ", clean_query)


    high_complexity_terms = [
        "compare", "analyze", "analyse", "explain", "why", "how",
        "tradeoff", "architecture", "design", "evaluate", "pros and cons"
    ]

    token_count = len(query.split())

    if any(term in query for term in high_complexity_terms):
        complexity = "high"
    elif token_count <= 4:
        complexity = "low"
    else:
        complexity = "medium"

    debug_info = state.get("debug_info", [])
    debug_info.append(f"complexity={complexity}")

    logger.info("Complexity classified as %s for query=%r", complexity, state["query"])

    
    logger.info("query='%s'", query)
    logger.info("match_terms=%s", [
        term for term in high_complexity_terms if term in query
    ])


    return {
        "complexity": complexity,
        "debug_info": debug_info,
    }

# -----------------------------
# NODE 5: Model selection node
# -----------------------------

def select_model_node(state: GraphState) -> GraphState:
    complexity = state["complexity"]
    config = MODEL_POLICY.get(complexity, MODEL_POLICY["medium"])

    model_alias = config["primary"]
    fallback_model_alias = config["fallback"]

    debug_info = state.get("debug_info", [])
    debug_info.append(f"model_alias={model_alias}")
    debug_info.append(f"fallback_model_alias={fallback_model_alias}")

    logger.info(
        "Selected model_alias=%s fallback_model_alias=%s for complexity=%s",
        model_alias,
        fallback_model_alias,
        complexity
    )

    return {
        "model_alias": model_alias,
        "fallback_model_alias": fallback_model_alias,
        "debug_info": debug_info,
    }



# -----------------------------
# NODE 6: generate answer
# -----------------------------
def generate_node(state: GraphState) -> GraphState:
    from rag.llm import generate_answer

    docs = state.get("docs", [])

    if not docs:
        return {"answer": "I don't know"}

    primary_model = state.get("model_alias", "nvidia-balanced")
    fallback_model = state.get("fallback_model_alias", "nvidia-fast")
    debug_info = state.get("debug_info", [])

    try:
        answer = generate_answer(
            query=state["query"],
            docs=docs,
            role=state["role"],
            model=primary_model,
        )
        debug_info.append(f"model_used={primary_model}")

        return {
            "answer": answer,
            "debug_info": debug_info,
        }

    except Exception as e:
        logger.warning(
            "Primary model failed: %s. Falling back to %s",
            primary_model,
            fallback_model,
        )
        debug_info.append(f"primary_failed={primary_model}")
        debug_info.append(f"fallback_to={fallback_model}")

        answer = generate_answer(
            query=state["query"],
            docs=docs,
            role=state["role"],
            model=fallback_model,
        )

        debug_info.append(f"model_used={fallback_model}")

        return {
            "answer": answer,
            "debug_info": debug_info,
        }



# -----------------------------
# BUILD GRAPH
# -----------------------------
def build_graph():
    graph = StateGraph(GraphState)

    # ✅ Nodes
    graph.add_node("classify", classify_query_node)
    graph.add_node("access_check", access_check_node)
    graph.add_node("deny", deny_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("complexity", classify_complexity_node)
    graph.add_node("model_select", select_model_node)
    graph.add_node("generate", generate_node)

    # ✅ Entry point
    graph.set_entry_point("classify")

    # ✅ Classification → access
    graph.add_edge("classify", "access_check")

    # ✅ Access routing
    graph.add_conditional_edges(
        "access_check",
        access_router,
        {
            "deny": "deny",
            "retrieve": "retrieve",
        }
    )

    # ✅ Deny ends
    graph.add_edge("deny", END)

    # ✅ ✅ Correct flow after retrieval (THIS WAS MISSING)
    graph.add_edge("retrieve", "complexity")
    graph.add_edge("complexity", "model_select")
    graph.add_edge("model_select", "generate")

    # ✅ End
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
