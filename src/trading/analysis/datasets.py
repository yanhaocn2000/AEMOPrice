"""Utility dataset generators for strategy experimentation."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple


def multi_strategy_benchmark_dataset(
    length: int = 120,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Synthetic dataset with alternating regimes for multi-strategy tests."""

    start = datetime(2024, 1, 1)
    market_data: List[Dict[str, object]] = []
    feature_data: List[Dict[str, object]] = []
    price = 120.0

    for idx in range(length):
        timestamp = start + timedelta(hours=1) * idx
        phase = idx % 12
        if phase < 6:
            actual_return = 0.012
        elif phase < 9:
            actual_return = -0.004
        else:
            actual_return = 0.018
        predicted_return = actual_return * 0.85 + 0.002

        price *= 1 + actual_return
        market_data.append({"timestamp": timestamp.isoformat(), "price": price, "region": "NSW1"})
        feature_data.append(
            {
                "timestamp": timestamp.isoformat(),
                "predicted_return": predicted_return,
                "target_future_return": actual_return,
            }
        )

    return market_data, feature_data


def high_growth_dataset(
    length: int = 240,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Dataset biased toward strong positive returns for profitability studies."""

    start = datetime(2024, 1, 1)
    market_data: List[Dict[str, object]] = []
    feature_data: List[Dict[str, object]] = []
    price = 150.0

    for idx in range(length):
        timestamp = start + timedelta(hours=1) * idx
        seasonal = (idx % 24) / 24
        if idx % 24 < 10:
            base_return = 0.014 + 0.002 * seasonal
        elif idx % 24 < 16:
            base_return = 0.006 - 0.001 * seasonal
        else:
            base_return = 0.02 + 0.001 * seasonal
        shock = 0.002 if idx % 7 == 0 else 0.0
        actual_return = base_return + shock
        predicted_return = actual_return * 0.9 + 0.003

        price *= 1 + actual_return
        market_data.append({"timestamp": timestamp.isoformat(), "price": price, "region": "NSW1"})
        feature_data.append(
            {
                "timestamp": timestamp.isoformat(),
                "predicted_return": predicted_return,
                "target_future_return": actual_return,
            }
        )

    return market_data, feature_data

