"""
Refund flow: eligibility check → human approval (graph pauses) → execution.
"""

from langgraph.types import interrupt

from src.graph.state import AgentState
from src.orders.db import get_order_status
from src.orders.refunds import check_refund_eligibility, process_refund, today


def refund_check_node(state: AgentState):
    """Step 1, eligibility: look up the order and apply the refund policy. No side effects."""
    order_id = state.get("order_id")
    if order_id is None:
        return {
            "llm_output": "I can help with a refund. What is your order number?",
            "refund_eligible": False,
            "answer_valid": True,
        }
    order = get_order_status.invoke({"order_id": order_id})
    eligible, reason = check_refund_eligibility(order, today())
    if order["status"] == "not_found":
        message = f"I couldn't find order {order_id}. Please check the number."
    elif not eligible:
        message = f"Order {order_id} isn't eligible for a refund: {reason}."
    else:
        message = (
            f"Order {order_id} is eligible for a refund of ₹{order['amount_inr']:,}. "
            "Your request is waiting for approval by a support agent."
        )
    return {
        "order": order,
        "refund_eligible": eligible,
        "refund_reason": reason,
        "llm_output": message,
        "answer_valid": True,
    }


def human_approval_node(state: AgentState):
    """Step 2, human approval: pause the graph until a support agent resumes it."""
    order = state["order"]
    decision = interrupt(
        {
            "action": "process_refund",
            "order_id": order["order_id"],
            "item": order["item"],
            "amount_inr": order["amount_inr"],
            "eligibility": state["refund_reason"],
        }
    )
    return {"refund_approved": bool(decision.get("approved"))}


def process_refund_node(state: AgentState):
    """Step 3, execution: runs only after approval."""
    result = process_refund.invoke({"order_id": state["order"]["order_id"]})
    if result["status"] != "refunded":
        message = f"Sorry, the refund could not be processed: {result['reason']}."
    else:
        message = (
            f"Your refund of ₹{result['amount_inr']:,} for order {result['order_id']} "
            f"has been processed. Reference: {result['reference']}."
        )
    return {"refund": result, "llm_output": message, "answer_valid": True}


def refund_rejected_node(state: AgentState):
    """Approval was declined: no refund is executed."""
    return {
        "llm_output": (
            f"We're sorry, but after review we can't approve a refund for order "
            f"{state['order']['order_id']}. A support agent will follow up with details."
        ),
        "answer_valid": True,
    }
