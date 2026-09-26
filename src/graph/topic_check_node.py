from functools import lru_cache
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.graph.state import AgentState
from src.graph.utils import get_llm


# Structured output for topic classification
class GradeTopic(BaseModel):
    """Structured output for topic classification."""

    score: Literal["Yes", "No"] = Field(
        description="Whether the question is about customer support."
    )


@lru_cache(maxsize=100)
def classify_topic(question: str) -> GradeTopic:
    system = """You are a grader assessing whether a user's question is related to customer support 
    for a product or a purchase.
    Customer support topics include:
    - Questions about purchasing products (e.g., "How do I place an order?")
    - Questions about order cancellations (e.g., "Can I cancel my order?")
    - Questions about refunds or returns (e.g., "How do I request a refund?")
    - Questions about product issues (e.g., "My product is not working.")
    - Questions about account issues (e.g., "I can't log in to my account.")

    If the question is about customer support, respond with "Yes". Otherwise, respond with "No".
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}"),
        ]
    )

    llm = get_llm()

    # Use structured output for better results
    structured_llm = llm.with_structured_output(GradeTopic)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({"question": question})
    return result


def topic_classifier(state: AgentState):
    """Classify the topic of the question."""
    question = state["question"]
    result = classify_topic(question)

    # Default to "on topic" if confidence is low
    # state["on_topic"] = result.score
    if result.score == "Yes":
        return {"on_topic": "Yes"}
    else:
        return {
            "on_topic": "No",
            "llm_output": "Please ask a question about customer support so I can help you better.",
        }
