"""
This module contains the FastAPI application that serves the RAG Graph API.
"""

import os
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse

from src.graph.graph import create_workflow
from src.graph.utils import load_faiss_index

warnings.filterwarnings("ignore")


class Question(BaseModel):
    question: str


api_context = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager to handle the lifespan events of the FastAPI application."""
    try:
        # Load the FAISS index
        faisss_index = load_faiss_index()
        # Create the workflow
        logger.info("Creating the workflow...")
        api_context["workflow"] = create_workflow(faisss_index)
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


@app.post("/answer")
async def answer(question: Question):
    """
    Answer the question.

    Args:
        question (Question): The question.

    Returns:
        JSONResponse: The response.
    """
    try:
        # Run the workflow
        graph = api_context["workflow"]
        # Graph nodes are sync (LLM Guard, FAISS); run off the event loop.
        state = await run_in_threadpool(graph.invoke, {"question": question.question})
        response = {
            "llm_output": state.get("llm_output"),
            "question_valid": state.get("question_valid"),
            "intent": state.get("intent"),
            "order": state.get("order"),
            "answer_valid": state.get("answer_valid", False),
        }
        logger.info(f"Response: {response}")
        return JSONResponse(content=response)
    except Exception:
        logger.exception("Failed to answer the question.")
        raise HTTPException(
            status_code=500,
            detail="Failed to answer the question.",
        )


@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"})
