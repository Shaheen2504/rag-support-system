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


