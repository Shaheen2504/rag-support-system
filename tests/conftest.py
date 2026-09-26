import pytest

from src.config import settings


@pytest.fixture(autouse=True)
def temp_orders_db(tmp_path, monkeypatch):
    """Each test gets a fresh seeded order DB; data/orders.db is never touched."""
    monkeypatch.setattr(settings, "ORDERS_DB_PATH", str(tmp_path / "orders.db"))
