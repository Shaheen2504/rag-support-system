"""
RAG System Evaluator

This module evaluates a RAG system using RAGAS metrics by sampling documents,
generating answers, and measuring performance.
"""

import os
import random
from uuid import uuid4

from langchain_openai import ChatOpenAI
from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity
from loguru import logger
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall

from src.config import settings
from src.graph.graph import create_workflow
from src.graph.utils import load_faiss_index


def setup_components():
    """Initialize all required components for RAG evaluation."""
    input_scanners = [PromptInjection(), TokenLimit(), Toxicity()]
    retriever = load_faiss_index()
    rag_app = create_workflow(retriever, input_scanners=input_scanners)

    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0, max_tokens=1000)
    evaluator_llm = LangchainLLMWrapper(llm)

    return retriever, rag_app, evaluator_llm


