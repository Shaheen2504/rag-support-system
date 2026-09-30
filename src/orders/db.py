"""Mock order backend: a small SQLite database seeded with synthetic orders."""

import sqlite3
from datetime import date, timedelta

from langchain_core.tools import tool

from src.config import settings

# Synthetic data only; no real customers. Dates are days relative to when the
# DB is first created, so the refund demo (1042 inside the 30-day window, 1047
# outside it) never goes stale.
SEED_ORDERS = [
    # order_id, item, amount_inr, status, order_date, expected_delivery
    (1042, "Smartphone X2", 24999, "delivered", -12, -7),
    (1043, "Wireless Earbuds", 3499, "shipped", -3, 3),
    (1044, "Laptop Stand", 1299, "processing", -1, 6),
    (1045, "USB-C Charger", 899, "cancelled", -10, None),
    (1046, "Smartwatch S", 8999, "out_for_delivery", -5, 0),
    (1047, "Bluetooth Speaker", 2799, "delivered", -50, -45),
]


def _day(offset: int | None) -> str | None:
    return None if offset is None else (date.today() + timedelta(days=offset)).isoformat()


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
    conn.execute(
        """CREATE TABLE IF NOT EXISTS refunds (
            refund_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL UNIQUE REFERENCES orders(order_id),
            amount_inr INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    conn.executemany(
        "INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?, ?, ?)",
        [(*o[:4], _day(o[4]), _day(o[5])) for o in SEED_ORDERS],
    )
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
