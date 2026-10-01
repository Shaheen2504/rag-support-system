"""
Graph-based workflow for the assistant.
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

# local imports
from src.graph.answer_check_node import (
    answer_check_node,
    check_language_same,
    check_relevance,
    check_toxicity,
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
from src.graph.refund_node import (
    human_approval_node,
    process_refund_node,
    refund_check_node,
    refund_rejected_node,
)
from src.graph.router_node import order_status_node, router_node
from src.graph.utils import load_faiss_index


def create_workflow(retriever, checkpointer=None):
    """Create a workflow. The refund path pauses for approval, which needs a checkpointer."""
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
        {True: "router", False: END},
    )
    workflow.add_node("router", router_node)
    workflow.add_conditional_edges(
        "router",
        lambda state: state["intent"],
        {
            "FAQ": "retrieve_docs",
            "ORDER": "order_status",
            "REFUND": "refund_check",
            "OFF_TOPIC": END,
        },
    )
    workflow.add_node("refund_check", refund_check_node)
    workflow.add_conditional_edges(
        "refund_check",
        lambda state: state["refund_eligible"],
        {True: "human_approval", False: END},
    )
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_conditional_edges(
        "human_approval",
        lambda state: state["refund_approved"],
        {True: "process_refund", False: "refund_rejected"},
    )
    workflow.add_node("process_refund", process_refund_node)
    workflow.add_node("refund_rejected", refund_rejected_node)
    workflow.add_edge("process_refund", END)
    workflow.add_edge("refund_rejected", END)
    workflow.add_node("order_status", order_status_node)
    workflow.add_edge("order_status", END)
    workflow.add_node("retrieve_docs", partial(retrieve, faiss_retriever=retriever))
    workflow.add_node("docs_grader", grade_documents_node)
    workflow.add_node("check_language_same", check_language_same)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("check_toxicity", check_toxicity)
    workflow.add_node("answer_check_node", answer_check_node)

    workflow.add_node("generate_answer", answer_node)

    workflow.add_edge(START, "scan_prompt_injection")
    workflow.add_edge(START, "scan_toxicity")
    workflow.add_edge(START, "scan_token_limit")
    workflow.add_edge("scan_prompt_injection", "question_check_node")
    workflow.add_edge("scan_toxicity", "question_check_node")
    workflow.add_edge("scan_token_limit", "question_check_node")
    workflow.add_edge("retrieve_docs", "docs_grader")
    workflow.add_edge("docs_grader", "generate_answer")
    workflow.add_edge("generate_answer", "check_language_same")
    workflow.add_edge("generate_answer", "check_relevance")
    workflow.add_edge("generate_answer", "check_toxicity")
    workflow.add_edge("check_language_same", "answer_check_node")
    workflow.add_edge("check_relevance", "answer_check_node")
    workflow.add_edge("check_toxicity", "answer_check_node")
    workflow.add_edge("answer_check_node", END)

    graph = workflow.compile(checkpointer=checkpointer)
    return graph



def make_graph():
    """Entry point for `langgraph dev` (see langgraph.json)."""
    return create_workflow(load_faiss_index())


if __name__ == "__main__":
    graph = make_graph()
    for q in ["What is the capital of France?", "I wnat to return a package"]:
        print(graph.invoke({"question": q})["llm_output"])
