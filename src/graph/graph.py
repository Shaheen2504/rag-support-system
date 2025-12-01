"""
Graph-based workflow for the assistant.
"""

from functools import partial

from langchain.globals import set_debug
from langgraph.graph import END, START, StateGraph

# local imports
from src.graph.answer_check_node import (
    answer_check_node,
    check_language_same,
    check_relevance,
    check_sentiment,
)
from src.graph.answer_node import answer_node
from src.graph.docs_grader_node import grade_documents_node
from src.graph.question_check_node import (
    question_check_node,
    scan_prompt_injection,
    scan_token_limit,
    scan_toxicity,
)
from src.graph.retriever_node import retrieve
from src.graph.state import AgentState
from src.graph.topic_check_node import topic_classifier
from src.graph.utils import load_faiss_index

set_debug(True)


def create_workflow(retriever):
    """Create a workflow."""
    workflow = StateGraph(AgentState)
