from src.orders.db import get_order_status


def test_known_order():
    result = get_order_status.invoke({"order_id": 1042})
    assert result["status"] == "delivered"
    assert result["item"] == "Smartphone X2"


def test_unknown_order():
    assert get_order_status.invoke({"order_id": 9999})["status"] == "not_found"
