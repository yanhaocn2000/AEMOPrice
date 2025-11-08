"""Collection of rule-based strategies for comparison."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Mapping, Sequence

from src.trading.strategies.base import TradeSignal, TradingStrategy


class MeanReversionStrategy(TradingStrategy):
    """Buy when price deviates below the recent mean and sell when above."""

    def __init__(
        self,
        region: str,
        lookback: int = 12,
        entry_zscore: float = 1.0,
        lot_size: float = 5.0,
    ) -> None:
        super().__init__(region)
        self.lookback = max(2, lookback)
        self.entry_zscore = max(0.1, entry_zscore)
        self.lot_size = max(0.1, lot_size)
        self._history: List[float] = []

    def generate_signals(
        self,
        market_row: Mapping[str, Any],
        feature_row: Mapping[str, Any],
    ) -> Sequence[TradeSignal]:
        price = market_row.get("price")
        if price is None:
            return []

        timestamp = market_row.get("timestamp") or feature_row.get("timestamp")
        timestamp_dt = self._ensure_timestamp(timestamp)
        price = float(price)
        self._history.append(price)
        if len(self._history) < self.lookback:
            return []

        recent = self._history[-self.lookback :]
        mean = sum(recent) / len(recent)
        variance = sum((p - mean) ** 2 for p in recent) / len(recent)
        std_dev = variance ** 0.5
        if std_dev == 0:
            return []

        zscore = (price - mean) / std_dev
        if zscore <= -self.entry_zscore:
            return [TradeSignal(timestamp_dt, self.region, "BUY", self.lot_size, price)]
        if zscore >= self.entry_zscore:
            return [TradeSignal(timestamp_dt, self.region, "SELL", self.lot_size, price)]
        return []

    def _ensure_timestamp(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))


class MomentumStrategy(TradingStrategy):
    """Trade in the direction of recent price momentum."""

    def __init__(
        self,
        region: str,
        window: int = 6,
        return_threshold: float = 0.001,
        lot_size: float = 5.0,
    ) -> None:
        super().__init__(region)
        self.window = max(2, window)
        self.return_threshold = max(0.0, return_threshold)
        self.lot_size = max(0.1, lot_size)
        self._history: List[float] = []

    def generate_signals(
        self,
        market_row: Mapping[str, Any],
        feature_row: Mapping[str, Any],
    ) -> Sequence[TradeSignal]:
        price = market_row.get("price")
        if price is None:
            return []
        price = float(price)

        timestamp = market_row.get("timestamp") or feature_row.get("timestamp")
        timestamp_dt = self._ensure_timestamp(timestamp)

        previous_price = self._history[-1] if self._history else None
        self._history.append(price)
        if len(self._history) < self.window:
            return []

        returns = []
        for idx in range(1, min(len(self._history), self.window)):
            prev = self._history[-idx - 1]
            curr = self._history[-idx]
            if prev != 0:
                returns.append((curr - prev) / prev)

        if not returns:
            return []

        avg_return = sum(returns) / len(returns)
        if avg_return > self.return_threshold and previous_price is not None:
            return [TradeSignal(timestamp_dt, self.region, "BUY", self.lot_size, price)]
        if avg_return < -self.return_threshold and previous_price is not None:
            return [TradeSignal(timestamp_dt, self.region, "SELL", self.lot_size, price)]
        return []

    def _ensure_timestamp(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))


class VolatilityBreakoutStrategy(TradingStrategy):
    """Enter trades when the price escapes recent volatility bands."""

    def __init__(
        self,
        region: str,
        lookback: int = 10,
        volatility_multiplier: float = 1.5,
        lot_size: float = 5.0,
    ) -> None:
        super().__init__(region)
        self.lookback = max(3, lookback)
        self.volatility_multiplier = max(0.5, volatility_multiplier)
        self.lot_size = max(0.1, lot_size)
        self._history: List[float] = []

    def generate_signals(
        self,
        market_row: Mapping[str, Any],
        feature_row: Mapping[str, Any],
    ) -> Sequence[TradeSignal]:
        price = market_row.get("price")
        if price is None:
            return []
        price = float(price)

        timestamp = market_row.get("timestamp") or feature_row.get("timestamp")
        timestamp_dt = self._ensure_timestamp(timestamp)

        previous_price = self._history[-1] if self._history else price
        self._history.append(price)
        if len(self._history) < self.lookback:
            return []

        recent = self._history[-self.lookback :]
        avg_range = self._average_absolute_change(recent)
        upper_band = max(recent) + avg_range * self.volatility_multiplier
        lower_band = min(recent) - avg_range * self.volatility_multiplier

        if price > upper_band and price > previous_price:
            return [TradeSignal(timestamp_dt, self.region, "BUY", self.lot_size, price)]
        if price < lower_band and price < previous_price:
            return [TradeSignal(timestamp_dt, self.region, "SELL", self.lot_size, price)]
        return []

    def _average_absolute_change(self, values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0
        diffs = [abs(values[idx] - values[idx - 1]) for idx in range(1, len(values))]
        return sum(diffs) / len(diffs)

    def _ensure_timestamp(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))


class AdaptiveThresholdStrategy(TradingStrategy):
    """Adjust position sizing based on signal strength to fight low returns."""

    def __init__(
        self,
        region: str,
        base_threshold: float = 0.002,
        risk_aversion: float = 4000.0,
        max_lot: float = 20.0,
    ) -> None:
        super().__init__(region)
        self.base_threshold = max(0.0, base_threshold)
        self.risk_aversion = max(1.0, risk_aversion)
        self.max_lot = max(1.0, max_lot)

    def generate_signals(
        self,
        market_row: Mapping[str, Any],
        feature_row: Mapping[str, Any],
    ) -> Sequence[TradeSignal]:
        predicted_return = feature_row.get("predicted_return")
        price = market_row.get("price")
        timestamp = market_row.get("timestamp") or feature_row.get("timestamp")
        if predicted_return is None or price is None or timestamp is None:
            return []

        timestamp_dt = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp))
        strength = abs(float(predicted_return))
        if strength < self.base_threshold:
            return []

        direction = "BUY" if predicted_return > 0 else "SELL"
        dynamic_size = min(self.max_lot, max(1.0, strength * self.risk_aversion))
        return [TradeSignal(timestamp_dt, self.region, direction, dynamic_size, float(price))]

