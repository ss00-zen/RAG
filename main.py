from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from auth.auth import verify_token, get_auth_provider
from auth.auth_context import normalize_user, role_from_auth
from rag.orchestrator import orchestrate, set_runtime_objects


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
    messages: List[Dict] = Field(default_factory=list)


def _extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = parts[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")

    return token


@app.post("/chat")
def chat(request: ChatRequest, authorization: Optional[str] = Header(default=None)):
    token = _extract_bearer_token(authorization)

    try:
        claims = verify_token(token)
        auth_ctx = normalize_user(claims, provider=get_auth_provider())
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")

    role = role_from_auth(auth_ctx)

    result = orchestrate(
        query=request.query,
        auth_ctx=auth_ctx,                  # keeps current orchestrator signature unchanged
        user_id=auth_ctx.user_id,
        messages=request.messages,
    )

    return {
        "answer": result.get("answer", "No answer"),
        "classification": result.get("classification"),
        "allowed": result.get("allowed"),
        "debug_info": result.get("debug_info", []),
        "user_id": auth_ctx.user_id,
        "role": role,
        "provider": auth_ctx.provider,
    }
