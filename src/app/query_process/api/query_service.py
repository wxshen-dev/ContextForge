from pathlib import Path
from uuid import uuid4

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.clients.mongo_history_utils import clear_history, get_recent_messages
from app.core.logger import logger
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    set_task_result,
    update_task_status,
)

app = FastAPI(title="ContextForge Query Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to answer")
    session_id: str | None = Field(None, description="Existing conversation ID")
    is_stream: bool = Field(False, description="Send answer events over SSE")


@app.get("/chat.html")
async def chat() -> FileResponse:
    page = Path(__file__).resolve().parent.parent / "page" / "chat.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="Chat page not found")
    return FileResponse(page)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


def run_query_graph(session_id: str, user_query: str, is_stream: bool = True) -> dict | None:
    state = create_query_default_state(
        original_query=user_query,
        session_id=session_id,
        is_stream=is_stream,
    )
    try:
        result = query_app.invoke(state)
        answer = result.get("answer", "")
        if answer:
            set_task_result(session_id, "answer", answer)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
        return result
    except Exception:
        logger.exception("Query workflow failed for session {}", session_id)
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": "Query processing failed"})
        return None


@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest) -> dict:
    session_id = request.session_id or str(uuid4())
    if request.is_stream:
        create_sse_queue(session_id)
    update_task_status(session_id, TASK_STATUS_PROCESSING, request.is_stream)

    if request.is_stream:
        background_tasks.add_task(run_query_graph, session_id, request.query, True)
        return {"message": "Query processing started", "session_id": session_id}

    result = run_query_graph(session_id, request.query, False)
    if result is None:
        raise HTTPException(status_code=500, detail="Query processing failed")
    return {
        "message": "Query completed",
        "session_id": session_id,
        "answer": get_task_result(session_id, "answer", result.get("answer", "")),
        "image_urls": result.get("image_urls", []),
        "done_list": get_done_task_list(session_id),
    }


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request) -> StreamingResponse:
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/status/{session_id}")
async def status(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "status": get_task_status(session_id),
        "done_list": get_done_task_list(session_id),
        "running_list": get_running_task_list(session_id),
        "answer": get_task_result(session_id, "answer"),
    }


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50) -> dict:
    try:
        records = get_recent_messages(session_id, limit=limit)
    except Exception as exc:
        logger.exception("History lookup failed for session {}", session_id)
        raise HTTPException(status_code=500, detail="Could not load conversation history") from exc
    items = [
        {
            "_id": str(record.get("_id", "")),
            "session_id": record.get("session_id", ""),
            "role": record.get("role", ""),
            "text": record.get("text", ""),
            "rewritten_query": record.get("rewritten_query", ""),
            "item_names": record.get("item_names", []),
            "image_urls": record.get("image_urls", []),
            "ts": record.get("ts"),
        }
        for record in records
    ]
    return {"session_id": session_id, "items": items}


@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str) -> dict:
    try:
        count = clear_history(session_id)
    except Exception as exc:
        logger.exception("History deletion failed for session {}", session_id)
        raise HTTPException(status_code=500, detail="Could not clear conversation history") from exc
    return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
