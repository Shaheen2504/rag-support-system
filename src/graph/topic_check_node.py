from functools import lru_cache
from typing import Any, Dict, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import settings
from src.graph.state import AgentState


# Structured output for topic classification
class GradeTopic(BaseModel):
    """Structured output for topic classification."""

    score: Literal["Yes", "No"] = Field(
        description="Whether the question is about customer support."
    )


@lru_cache(maxsize=100)
