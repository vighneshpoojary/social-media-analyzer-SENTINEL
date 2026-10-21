"""Storage layer for data persistence (ClickHouse, queues, etc.)."""

from src.storage.clickhouse_client import ClickHouseStorageClient

__all__ = ["ClickHouseStorageClient"]
