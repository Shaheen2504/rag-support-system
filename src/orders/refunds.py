"""
Mock refunds over the synthetic order DB. No real payment API is called.

Two separate steps:
- check_refund_eligibility: pure policy check, no side effects.
- process_refund: the only code that executes a refund (writes to SQLite).
  The graph calls it only after human approval.
"""

from datetime import date, datetime

from langchain_core.tools import tool

from src.orders.db import get_connection

REFUND_WINDOW_DAYS = 30


def check_refund_eligibility(order: dict, today: date) -> tuple[bool, str]:
    """Return (eligible, reason) for a refund on `order` under the synthetic policy."""
    status = order["status"]
    if status == "not_found":
        return False, "the order does not exist"
    if status == "refunded":
        return False, "the order has already been refunded"
    if status == "cancelled":
        return False, "the order was cancelled, so there is no payment to refund"
    if status != "delivered":
        return False, "the order has not been delivered yet; you can cancel it instead"
    days = (today - date.fromisoformat(order["expected_delivery"])).days
    if days > REFUND_WINDOW_DAYS:
        return False, f"it was delivered {days} days ago, outside the {REFUND_WINDOW_DAYS}-day refund window"
    return True, f"delivered {days} days ago, within the {REFUND_WINDOW_DAYS}-day refund window"


@tool
def process_refund(order_id: int) -> dict:
    """Execute a (mock) refund for an order: record it and mark the order refunded."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        order = dict(row) if row else {"order_id": order_id, "status": "not_found"}
        eligible, reason = check_refund_eligibility(order, date.today())
        if not eligible:
            return {"order_id": order_id, "status": "refused", "reason": reason}
        cur = conn.execute(
            "INSERT INTO refunds (order_id, amount_inr, created_at) VALUES (?, ?, ?)",
            (order_id, order["amount_inr"], datetime.now().isoformat(timespec="seconds")),
        )
        conn.execute("UPDATE orders SET status = 'refunded' WHERE order_id = ?", (order_id,))
    return {
        "order_id": order_id,
        "status": "refunded",
        "amount_inr": order["amount_inr"],
        "reference": f"RF-{cur.lastrowid:05d}",
    }
