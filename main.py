from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional

from rag.orchestrator import orchestrate, set_runtime_objects

# IMPORTANT:
# replace these with your real load/init functions
# Example:
# from rag.bootstrap import load_runtime
# db, bm25, chunks = load_runtime()

app = FastAPI()

# ---------------------------------------------------
# TODO: replace with actual runtime initialization
# ---------------------------------------------------
db = ...
bm25 = ...
chunks = ...

set_runtime_objects(db, bm25, chunks)


class ChatRequest(BaseModel):
    query: str
    role: str
    username: Optional[str] = None
    messages: List[Dict] = []


@app.post("/chat")
def chat(request: ChatRequest):
    result = orchestrate(
        query=request.query,
        role=request.role,
        user_id=request.username,
        messages=request.messages,
    )

    return {
        "answer": result.get("answer", "No answer"),
        "classification": result.get("classification"),
        "allowed": result.get("allowed"),
        "debug_info": result.get("debug_info", []),
    }
