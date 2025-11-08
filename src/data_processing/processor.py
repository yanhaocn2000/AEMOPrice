"""Utilities for cleaning and merging AEMO datasets using built-in types."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List

from src.utils.logging_config import setup_logging


Row = Dict[str, Any]


class MarketDataProcessor:
    """Prepare market datasets for feature engineering and modelling."""

    def __init__(self) -> None:
        self.logger = setup_logging(self.__class__.__name__)

    def normalize_columns(self, rows: Iterable[Row]) -> List[Row]:
        """Normalize column names to snake_case for consistency."""

        normalized_rows: List[Row] = []
        for row in rows:
            normalized_rows.append({self._normalize_key(key): value for key, value in row.items()})
        return normalized_rows

    def merge_datasets(self, datasets: Dict[str, Iterable[Row]], on: List[str]) -> List[Row]:
        """Merge multiple datasets on a shared set of columns."""

        if not datasets:
            raise ValueError("No datasets provided for merging")

        merged: Dict[tuple, Row] = {}
        for name, dataset in datasets.items():
            self.logger.info("Merging dataset %s", name)
            normalized = self.normalize_columns(dataset)
            for row in normalized:
                key = tuple(row[field] for field in on)
                target = merged.setdefault(key, {field: row[field] for field in on})
                for column, value in row.items():
                    if column in on:
                        continue
                    suffix = f"_{name}" if column in target else ""
                    target[f"{column}{suffix}"] = value

        sorted_rows = [merged[key] for key in sorted(merged.keys())]
        return sorted_rows

    def fill_missing_values(self, rows: Iterable[Row], method: str = "ffill") -> List[Row]:
        """Fill missing values using a simple forward/backward fill strategy."""

        if method not in {"ffill", "bfill"}:
            raise ValueError("method must be either 'ffill' or 'bfill'")

        filled_rows: List[Row] = []
        last_values: Dict[str, Any] = {}
        rows_list = list(rows)
        indices = range(len(rows_list)) if method == "ffill" else range(len(rows_list) - 1, -1, -1)

        for idx in indices:
            row = dict(rows_list[idx])
            for key, value in row.items():
                if value is None:
                    if key in last_values:
                        row[key] = last_values[key]
                else:
                    last_values[key] = value
            filled_rows.append(row)

        if method == "ffill":
            return filled_rows
        return list(reversed(filled_rows))

    def add_time_features(self, rows: Iterable[Row], time_column: str) -> List[Row]:
        """Add simple time based features derived from the timestamp column."""

        enriched_rows: List[Row] = []
        for row in rows:
            timestamp_value = row.get(time_column)
            timestamp = self._ensure_datetime(timestamp_value)
            enriched_row = dict(row)
            enriched_row["hour"] = timestamp.hour
            enriched_row["day_of_week"] = timestamp.weekday()
            enriched_row["month"] = timestamp.month
            enriched_rows.append(enriched_row)
        return enriched_rows

    def _normalize_key(self, key: str) -> str:
        return key.strip().lower().replace(" ", "_")

    def _ensure_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))
