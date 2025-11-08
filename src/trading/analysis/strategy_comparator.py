"""Utilities to compare multiple strategies on the same dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from src.backtesting.backtest_engine import BacktestEngine
from src.trading.execution.order_executor import SimulatedOrderExecutor
from src.trading.strategies.base import TradingStrategy
from src.utils.logging_config import setup_logging


MarketRow = Mapping[str, object]
FeatureRow = Mapping[str, object]


@dataclass
class StrategyPerformance:
    name: str
    metrics: Dict[str, float]
    trade_count: int


@dataclass
class ComparisonReport:
    performances: List[StrategyPerformance]
    leader: StrategyPerformance | None


class StrategyComparator:
    """Run a backtest for each strategy and surface comparative metrics."""

    def __init__(self, initial_capital: float = 1_000_000.0) -> None:
        self.initial_capital = initial_capital
        self.logger = setup_logging(self.__class__.__name__)

    def compare(
        self,
        market_data: Sequence[MarketRow],
        feature_data: Sequence[FeatureRow],
        strategies: Iterable[TradingStrategy],
    ) -> ComparisonReport:
        results: List[StrategyPerformance] = []
        for strategy in strategies:
            self.logger.info("Backtesting strategy %s", strategy.__class__.__name__)
            executor = SimulatedOrderExecutor()
            engine = BacktestEngine(strategy=strategy, executor=executor, initial_capital=self.initial_capital)
            result = engine.run(market_data, feature_data)
            results.append(
                StrategyPerformance(
                    name=strategy.__class__.__name__,
                    metrics=result.metrics,
                    trade_count=len(result.trades),
                )
            )

        leader = max(results, key=lambda perf: perf.metrics.get("total_return", 0.0), default=None)
        return ComparisonReport(performances=results, leader=leader)


def summarise_report(report: ComparisonReport) -> List[Tuple[str, float, float, float, int]]:
    """Convert a comparison report to a tuple-based summary for reporting."""

    summary: List[Tuple[str, float, float, float, int]] = []
    for perf in report.performances:
        metrics = perf.metrics
        summary.append(
            (
                perf.name,
                float(metrics.get("total_return", 0.0)),
                float(metrics.get("sharpe", 0.0)),
                float(metrics.get("max_drawdown", 0.0)),
                perf.trade_count,
            )
        )
    return summary

