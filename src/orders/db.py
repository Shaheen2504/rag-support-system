"""Mock order backend: a small SQLite database seeded with synthetic orders."""

import sqlite3

from langchain_core.tools import tool

from src.config import settings

# Synthetic data only; no real customers.
SEED_ORDERS = [
    # order_id, item, amount_inr, status, order_date, expected_delivery
    (1042, "Smartphone X2", 24999, "delivered", "2026-09-18", "2026-09-23"),
    (1043, "Wireless Earbuds", 3499, "shipped", "2026-09-22", "2026-09-29"),
    (1044, "Laptop Stand", 1299, "processing", "2026-09-25", "2026-10-02"),
    (1045, "USB-C Charger", 899, "cancelled", "2026-09-20", None),
    (1046, "Smartwatch S", 8999, "out_for_delivery", "2026-09-21", "2026-09-27"),
    (1047, "Bluetooth Speaker", 2799, "delivered", "2026-08-10", "2026-08-15"),
]


def get_connection() -> sqlite3.Connection:
    """Open the orders DB, creating and seeding it on first use."""
    conn = sqlite3.connect(settings.ORDERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            item TEXT NOT NULL,
            amount_inr INTEGER NOT NULL,
            status TEXT NOT NULL,
            order_date TEXT NOT NULL,
            expected_delivery TEXT
        )"""
    )
    conn.executemany("INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?, ?, ?)", SEED_ORDERS)
    conn.commit()
    return conn


@tool
def get_order_status(order_id: int) -> dict:
    """Look up an order by its numeric ID and return its status and details."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
    return dict(row) if row else {"order_id": order_id, "status": "not_found"}
