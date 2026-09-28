"""
This module contains the FastAPI application that serves the RAG Graph API.
"""

import os
import secrets
import warnings
from uuid import uuid4
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.config import settings
from src.graph.graph import create_workflow
from src.graph.utils import load_faiss_index

warnings.filterwarnings("ignore")


class Question(BaseModel):
    question: str


class Approval(BaseModel):
    thread_id: str
    approved: bool


api_context = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager to handle the lifespan events of the FastAPI application."""
    try:
        # Load the FAISS index
        faisss_index = load_faiss_index()
        # Create the workflow
        logger.info("Creating the workflow...")
        # In-memory checkpointer: lets the refund path pause for approval.
        api_context["workflow"] = create_workflow(faisss_index, checkpointer=MemorySaver())
        yield
    except Exception:
        logger.exception("Failed to load FAISS index and create the workflow.")
        raise HTTPException(
            status_code=500,
            detail="Failed to load FAISS index and create the workflow.",
        )
    del faisss_index
    del api_context["workflow"]
    logger.info("Workflow deleted.")


app = FastAPI(title="Rag Graph API", version="0.1.0", lifespan=lifespan)


static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
def read_root():
    return FileResponse(static_path + "/index.html")


def build_response(graph, config) -> dict:
    """Answer fields from the thread's state, plus any approval the graph waits on."""
    snapshot = graph.get_state(config)
    state = snapshot.values
    return {
        "llm_output": state.get("llm_output"),
        "question_valid": state.get("question_valid"),
        "intent": state.get("intent"),
        "order": state.get("order"),
        "refund": state.get("refund"),
        "answer_valid": state.get("answer_valid", False),
        "thread_id": config["configurable"]["thread_id"],
        "pending_approval": snapshot.interrupts[0].value if snapshot.interrupts else None,
    }


@app.post("/answer")
async def answer(question: Question):
    """
    Answer the question. Each request runs in a new thread; if a refund needs
    approval, the response carries `pending_approval` and the `thread_id` to resume.
    """
    try:
        graph = api_context["workflow"]
        config = {"configurable": {"thread_id": str(uuid4())}}
        # Graph nodes are sync (LLM Guard, FAISS); run off the event loop.
        await run_in_threadpool(graph.invoke, {"question": question.question}, config)
        response = build_response(graph, config)
        logger.info(f"Response: {response}")
        return JSONResponse(content=response)
    except Exception:
        logger.exception("Failed to answer the question.")
        raise HTTPException(
            status_code=500,
            detail="Failed to answer the question.",
        )


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_approval_key(api_key: str | None = Depends(api_key_header)) -> None:
    """Only support agents holding APPROVAL_API_KEY may approve refunds. Fails closed if unset."""
    expected = settings.APPROVAL_API_KEY
    if (
        expected is None
        or api_key is None
        or not secrets.compare_digest(api_key, expected.get_secret_value())
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.post("/approve", dependencies=[Depends(require_approval_key)])
async def approve(approval: Approval):
    """Human (support agent) decision on a paused refund; resumes the graph."""
    graph = api_context["workflow"]
    config = {"configurable": {"thread_id": approval.thread_id}}
    if not graph.get_state(config).interrupts:
        raise HTTPException(status_code=404, detail="No refund awaiting approval.")
    await run_in_threadpool(
        graph.invoke, Command(resume={"approved": approval.approved}), config
    )
    response = build_response(graph, config)
    logger.info(f"Approval {approval}: {response}")
    return JSONResponse(content=response)


@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"})
