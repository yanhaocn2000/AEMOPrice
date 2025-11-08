"""Machine learning based trading strategy."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from src.trading.strategies.base import TradeSignal, TradingStrategy


class MLTradingStrategy(TradingStrategy):
    """Generate trading signals from model predictions."""

    def __init__(self, region: str, threshold: float = 0.01, lot_size: float = 1.0) -> None:
        super().__init__(region)
        self.threshold = threshold
        self.lot_size = lot_size

    def generate_signals(
        self,
        market_row: Mapping[str, Any],
        feature_row: Mapping[str, Any],
    ) -> Sequence[TradeSignal]:
        predicted_return = feature_row.get("predicted_return")
        current_price = market_row.get("price")

        if predicted_return is None or current_price is None:
            self.logger.debug("Missing data for signal generation")
            return []

        timestamp = market_row.get("timestamp") or feature_row.get("timestamp")
        if not isinstance(timestamp, datetime):
            try:
                timestamp = datetime.fromisoformat(str(timestamp))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                self.logger.debug("Unable to parse timestamp from row: %s", timestamp)
                return []

        if predicted_return > self.threshold:
            return [TradeSignal(timestamp, self.region, "BUY", self.lot_size, float(current_price))]
        if predicted_return < -self.threshold:
            return [TradeSignal(timestamp, self.region, "SELL", self.lot_size, float(current_price))]
        return []

