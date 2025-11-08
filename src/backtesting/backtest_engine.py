"""Backtesting engine implementing a simplified event-driven loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Sequence

from src.trading.execution.order_executor import ExecutionReport, SimulatedOrderExecutor
from src.trading.strategies.base import TradingStrategy
from src.utils.logging_config import setup_logging


MarketRow = Mapping[str, Any]
FeatureRow = Mapping[str, Any]


@dataclass
class Trade:
    timestamp: datetime
    strategy: str
    region: str
    action: str
    quantity: float
    price: float
    commission: float
    slippage: float
    pnl: float


@dataclass
class EquityPoint:
    timestamp: datetime
    value: float


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: List[EquityPoint]
    metrics: Dict[str, float]


class BacktestEngine:
    def __init__(
        self,
        strategy: TradingStrategy,
        executor: SimulatedOrderExecutor,
        initial_capital: float = 1_000_000.0,
    ) -> None:
        self.strategy = strategy
        self.executor = executor
        self.initial_capital = initial_capital
        self.logger = setup_logging(self.__class__.__name__)

    def run(self, market_data: Sequence[MarketRow], feature_data: Sequence[FeatureRow]) -> BacktestResult:
        capital = self.initial_capital
        equity_curve: List[EquityPoint] = []
        trades: List[Trade] = []

        feature_lookup = {
            self._ensure_datetime(row["timestamp"]): row for row in feature_data if "timestamp" in row
        }

        for market_row in sorted(market_data, key=lambda row: self._ensure_datetime(row["timestamp"])):
            timestamp = self._ensure_datetime(market_row["timestamp"])
            feature_row = feature_lookup.get(timestamp, {})

            signals = self.strategy.generate_signals(market_row, feature_row)
            reports = self.executor.execute(list(signals))

            for report in reports:
                pnl = self._calculate_trade_pnl(report, market_row, feature_row)
                capital += pnl
                trades.append(
                    Trade(
                        timestamp=timestamp,
                        strategy=self.strategy.__class__.__name__,
                        region=report.signal.region,
                        action=report.signal.action,
                        quantity=report.signal.quantity,
                        price=report.executed_price,
                        commission=report.commission,
                        slippage=report.slippage,
                        pnl=pnl,
                    )
                )

            equity_curve.append(EquityPoint(timestamp=timestamp, value=capital))

        metrics = self._calculate_metrics(equity_curve)
        return BacktestResult(trades=trades, equity_curve=equity_curve, metrics=metrics)

    def _ensure_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))

    def _calculate_trade_pnl(
        self,
        report: ExecutionReport,
        market_row: MarketRow,
        feature_row: FeatureRow,
    ) -> float:
        actual_return = feature_row.get("target_future_return")
        base_price = market_row.get("price")

        if actual_return is None or base_price is None:
            return -report.commission

        quantity = float(report.signal.quantity)
        entry_price = float(report.executed_price)
        base_price = float(base_price)
        actual_return = float(actual_return)
        exit_price = base_price * (1 + actual_return)

        if report.signal.action == "BUY":
            gross_profit = (exit_price - entry_price) * quantity
        else:
            gross_profit = (entry_price - exit_price) * quantity

        return gross_profit - report.commission

    def _calculate_metrics(self, equity_curve: Sequence[EquityPoint]) -> Dict[str, float]:
        if len(equity_curve) < 2:
            return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

        values = [point.value for point in equity_curve]
        returns = []
        for idx in range(1, len(values)):
            previous = values[idx - 1]
            current = values[idx]
            returns.append(((current - previous) / previous) if previous != 0 else 0.0)

        average_return = sum(returns) / len(returns)
        variance = sum((r - average_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        sharpe = (average_return / volatility * (252 ** 0.5)) if volatility != 0 else 0.0
        total_return = (values[-1] / values[0]) - 1 if values[0] != 0 else 0.0
        max_drawdown = self._calculate_max_drawdown(values)
        return {
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
        }

    def _calculate_max_drawdown(self, values: Sequence[float]) -> float:
        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak if peak != 0 else 0.0
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        return float(max_drawdown)

