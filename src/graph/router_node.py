from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.graph.state import AgentState
from src.graph.utils import get_llm
from src.orders.db import get_order_status


class RouteQuestion(BaseModel):
    """Structured output for intent routing."""

    intent: Literal["FAQ", "ORDER", "OFF_TOPIC"] = Field(
        description="Which handler should answer the question."
    )
    order_id: Optional[int] = Field(
        default=None, description="Numeric order ID if the user gave one."
    )


system = """You route customer messages for an online store's support bot.

- ORDER: the user asks about the status, location or delivery of a specific order
  of theirs (e.g. "Where is my order 1042?", "Has my order shipped yet?").
- FAQ: any other customer-support question: accounts, passwords, payments,
  shipping options, cancellations, refunds, returns, invoices, complaints.
- OFF_TOPIC: anything not about the store or the user's purchases.

If the user mentions an order number, put it in order_id."""

route_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "User message: {question}")]
)


def router_node(state: AgentState):
    """Classify the question into FAQ / ORDER / OFF_TOPIC."""
    router = route_prompt | get_llm().with_structured_output(RouteQuestion)
    result = router.invoke({"question": state["question"]})
    update = {"intent": result.intent, "order_id": result.order_id}
    if result.intent == "OFF_TOPIC":
        update["llm_output"] = (
            "Please ask a question about customer support so I can help you better."
        )
    return update


STATUS_TEXT = {
    "processing": "is being processed and should arrive by {expected_delivery}",
    "shipped": "has shipped and should arrive by {expected_delivery}",
    "out_for_delivery": "is out for delivery and should arrive by {expected_delivery}",
    "delivered": "was delivered on {expected_delivery}",
    "cancelled": "was cancelled",
}


def order_status_node(state: AgentState):
    """
    Answer order questions from the order DB via the get_order_status tool.
    Replies are templates over DB fields (no LLM text), so they skip the
    output scanners and count as valid.
    """
    order_id = state.get("order_id")
    if order_id is None:
        return {
            "llm_output": "Sure, I can check that. What is your order number?",
            "answer_valid": True,
        }
    order = get_order_status.invoke({"order_id": order_id})
    if order["status"] == "not_found":
        return {
            "llm_output": f"I couldn't find order {order_id}. Please check the number.",
            "order": order,
            "answer_valid": True,
        }
    text = STATUS_TEXT[order["status"]].format(**order)
    return {
        "llm_output": f"Your order {order_id} ({order['item']}) {text}.",
        "order": order,
        "answer_valid": True,
    }
