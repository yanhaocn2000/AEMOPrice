"""Advanced trading strategies with distinct signal generation logic."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Deque, Mapping, Sequence

from src.trading.strategies.base import TradeSignal, TradingStrategy


class DualMovingAverageStrategy(TradingStrategy):
    """Trend-following strategy based on dual moving averages."""

    def __init__(
        self,
        region: str,
        short_window: int = 8,
        long_window: int = 24,
        threshold: float = 0.001,
        base_lot: float = 20.0,
        leverage: float = 8.0,
        max_lot: float = 800.0,
    ) -> None:
        super().__init__(region)
        long_window = max(3, long_window)
        short_window = max(2, min(short_window, long_window - 1))
        self.short_window = short_window
        self.long_window = long_window
        self.threshold = max(0.0, threshold)
        self.base_lot = max(1.0, base_lot)
        self.leverage = max(1.0, leverage)
        self.max_lot = max(self.base_lot, max_lot)
        self.prices: Deque[float] = deque(maxlen=self.long_window)

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
        self.prices.append(price)
        if len(self.prices) < self.long_window:
            return []

        short_prices = list(self.prices)[-self.short_window :]
        short_avg = sum(short_prices) / len(short_prices)
        long_avg = sum(self.prices) / len(self.prices)
        if long_avg == 0:
            return []

        strength = (short_avg - long_avg) / long_avg
        if abs(strength) < self.threshold:
            return []

        direction = "BUY" if strength > 0 else "SELL"
        dynamic_size = min(
            self.max_lot,
            max(self.base_lot, self.base_lot + abs(strength) * self.leverage * self.base_lot),
        )
        return [TradeSignal(timestamp_dt, self.region, direction, float(dynamic_size), price)]

    def _ensure_timestamp(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))


class CarryMomentumStrategy(TradingStrategy):
    """Blend of carry signal and momentum strength."""

    def __init__(
        self,
        region: str,
        carry_threshold: float = 0.001,
        momentum_weight: float = 2.0,
        base_lot: float = 25.0,
        risk_multiplier: float = 600.0,
        lookback: int = 12,
        max_lot: float = 900.0,
    ) -> None:
        super().__init__(region)
        self.carry_threshold = max(0.0, carry_threshold)
        self.momentum_weight = max(0.0, momentum_weight)
        self.base_lot = max(1.0, base_lot)
        self.risk_multiplier = max(1.0, risk_multiplier)
        self.max_lot = max(self.base_lot, max_lot)
        self.returns: Deque[float] = deque(maxlen=max(3, lookback))

    def generate_signals(
        self,
        market_row: Mapping[str, Any],
        feature_row: Mapping[str, Any],
    ) -> Sequence[TradeSignal]:
        price = market_row.get("price")
        predicted_return = feature_row.get("predicted_return")
        actual_return = feature_row.get("target_future_return")
        timestamp = market_row.get("timestamp") or feature_row.get("timestamp")

        if price is None or predicted_return is None or timestamp is None:
            return []

        if actual_return is not None:
            self.returns.append(float(actual_return))

        momentum = sum(self.returns) / len(self.returns) if self.returns else 0.0
        carry_component = float(predicted_return) - self.carry_threshold
        blended_signal = carry_component + self.momentum_weight * momentum
        if abs(blended_signal) <= 1e-9:
            return []

        timestamp_dt = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp))
        direction = "BUY" if blended_signal > 0 else "SELL"
        intensity = abs(blended_signal)
        dynamic_size = min(
            self.max_lot,
            max(self.base_lot, self.base_lot + intensity * self.risk_multiplier),
        )
        return [TradeSignal(timestamp_dt, self.region, direction, float(dynamic_size), float(price))]


class RegimeSwitchingStrategy(TradingStrategy):
    """Switch between momentum and mean-reversion based on volatility."""

    def __init__(
        self,
        region: str,
        volatility_window: int = 18,
        calm_threshold: float = 0.005,
        trend_threshold: float = 0.001,
        reversion_threshold: float = 0.002,
        base_lot: float = 30.0,
        max_lot: float = 1000.0,
    ) -> None:
        super().__init__(region)
        window = max(4, volatility_window)
        self.price_history: Deque[float] = deque(maxlen=window + 1)
        self.return_history: Deque[float] = deque(maxlen=window)
        self.calm_threshold = max(0.0, calm_threshold)
        self.trend_threshold = max(0.0, trend_threshold)
        self.reversion_threshold = max(0.0, reversion_threshold)
        self.base_lot = max(1.0, base_lot)
        self.max_lot = max(self.base_lot, max_lot)

    def generate_signals(
        self,
        market_row: Mapping[str, Any],
        feature_row: Mapping[str, Any],
    ) -> Sequence[TradeSignal]:
        price = market_row.get("price")
        predicted_return = feature_row.get("predicted_return")
        actual_return = feature_row.get("target_future_return")
        timestamp = market_row.get("timestamp") or feature_row.get("timestamp")

        if price is None or predicted_return is None or timestamp is None:
            return []

        price = float(price)
        timestamp_dt = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp))
        self.price_history.append(price)
        if len(self.price_history) >= 2:
            last_price = self.price_history[-2]
            if last_price != 0:
                self.return_history.append((price - last_price) / last_price)
        if actual_return is not None:
            self.return_history.append(float(actual_return))

        if len(self.return_history) < 3:
            return []

        avg_return = sum(self.return_history) / len(self.return_history)
        variance = sum((r - avg_return) ** 2 for r in self.return_history) / len(self.return_history)
        volatility = variance ** 0.5

        strength = float(predicted_return)
        if volatility <= self.calm_threshold:
            if abs(strength) < self.trend_threshold:
                return []
            direction = "BUY" if strength > 0 else "SELL"
        else:
            if abs(strength) < self.reversion_threshold:
                return []
            direction = "SELL" if strength > 0 else "BUY"
            strength = -strength

        intensity = abs(strength)
        dynamic_size = min(
            self.max_lot,
            max(self.base_lot, self.base_lot + intensity * self.base_lot * 10),
        )
        return [TradeSignal(timestamp_dt, self.region, direction, float(dynamic_size), price)]

