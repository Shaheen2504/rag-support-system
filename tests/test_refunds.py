from datetime import date

from src.orders.db import get_order_status
from src.orders.refunds import check_refund_eligibility, process_refund

TODAY = date(2026, 9, 27)


def order(status, delivered="2026-09-23"):
    return {"order_id": 1, "status": status, "expected_delivery": delivered}


def test_eligible_when_delivered_within_window():
    assert check_refund_eligibility(order("delivered"), TODAY)[0]


def test_window_boundary():
    assert check_refund_eligibility(order("delivered", "2026-08-28"), TODAY)[0]  # 30 days
    assert not check_refund_eligibility(order("delivered", "2026-08-27"), TODAY)[0]  # 31 days


def test_ineligible_statuses():
    for status in ["shipped", "processing", "out_for_delivery", "cancelled", "refunded", "not_found"]:
        eligible, reason = check_refund_eligibility(order(status), TODAY)
        assert not eligible, status
        assert reason


def test_process_refund_marks_order_refunded():
    result = process_refund.invoke({"order_id": 1042})
    assert result["status"] == "refunded"
    assert result["amount_inr"] == 24999
    assert result["reference"].startswith("RF-")
    assert get_order_status.invoke({"order_id": 1042})["status"] == "refunded"


def test_process_refund_is_not_repeatable():
    process_refund.invoke({"order_id": 1042})
    assert process_refund.invoke({"order_id": 1042})["status"] == "refused"


def test_process_refund_refuses_ineligible_and_unknown_orders():
    assert process_refund.invoke({"order_id": 1043})["status"] == "refused"  # shipped
    assert process_refund.invoke({"order_id": 9999})["status"] == "refused"
    assert get_order_status.invoke({"order_id": 1043})["status"] == "shipped"
