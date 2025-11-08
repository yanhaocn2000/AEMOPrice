"""Feature engineering pipeline built on standard Python data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.utils.logging_config import setup_logging


FeatureRow = Dict[str, Any]


@dataclass
class FeatureConfig:
    target_column: str
    lag_periods: List[int]
    rolling_windows: List[int]


class FeatureEngineer:
    """Create model ready features from market datasets without external deps."""

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.logger = setup_logging(self.__class__.__name__)

    def transform(self, rows: List[FeatureRow]) -> List[FeatureRow]:
        if not rows:
            return []

        target = self.config.target_column
        sorted_rows = sorted(rows, key=lambda row: row["timestamp"])
        prices = [float(row[target]) for row in sorted_rows]
        max_lag = max(self.config.lag_periods, default=0)

        transformed: List[FeatureRow] = []
        for idx in range(len(sorted_rows) - 1):  # exclude last row for future return
            if idx < max_lag:
                continue

            base_row = dict(sorted_rows[idx])

            for lag in self.config.lag_periods:
                base_row[f"{target}_lag_{lag}"] = prices[idx - lag]

            for window in self.config.rolling_windows:
                start = max(0, idx - window + 1)
                window_values = prices[start : idx + 1]
                base_row[f"{target}_rolling_mean_{window}"] = sum(window_values) / len(window_values)

            current_price = prices[idx]
            future_price = prices[idx + 1]
            base_row["target_future_return"] = (
                (future_price - current_price) / current_price if current_price != 0 else 0.0
            )

            transformed.append(base_row)

        self.logger.debug("Generated %d feature rows", len(transformed))
        return transformed
