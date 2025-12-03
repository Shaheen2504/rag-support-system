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
    workflow.add_node(
        "scan_prompt_injection",
        scan_prompt_injection,
    )
    workflow.add_node(
        "scan_toxicity",
        scan_toxicity,
    )
    workflow.add_node(
        "scan_token_limit",
        scan_token_limit,
    )
    workflow.add_node("question_check_node", question_check_node)
    workflow.add_conditional_edges(
        "question_check_node",
        lambda state: state["question_valid"],
        {True: "topic_classifier", False: END},
    )
    workflow.add_node("topic_classifier", topic_classifier)
    workflow.add_conditional_edges(
        "topic_classifier",
        lambda state: state["on_topic"],
        {
            "Yes": "retrieve_docs",
            "No": END,
        },
    )
    workflow.add_node("retrieve_docs", partial(retrieve, faiss_retriever=retriever))
    workflow.add_node("docs_grader", grade_documents_node)
    workflow.add_node("check_language_same", check_language_same)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("check_sentiment", check_sentiment)
    workflow.add_node("answer_check_node", answer_check_node)

    workflow.add_node("generate_answer", answer_node)

