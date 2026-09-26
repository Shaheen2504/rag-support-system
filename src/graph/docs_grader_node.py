from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.graph.state import AgentState
from src.graph.utils import get_llm


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Retrieved document: \n\n {document} \n\n User question: {question}",
        ),
    ]
)


def grade_documents_node(state: AgentState):
    """Keep only documents the LLM grades as relevant (graded in parallel)."""
    docs = state["documents"]
    question = state["question"]
    grader = grade_prompt | get_llm().with_structured_output(GradeDocuments)
    grades = grader.batch([{"question": question, "document": doc} for doc in docs])
    filtered_docs = [
        doc
        for doc, grade in zip(docs, grades)
        if grade.binary_score.strip().strip(".").lower() == "yes"
    ]
    return {"documents": filtered_docs}
