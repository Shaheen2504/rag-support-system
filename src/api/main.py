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

