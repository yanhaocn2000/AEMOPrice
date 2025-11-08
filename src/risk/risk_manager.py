"""Risk management utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src.trading.execution.order_executor import ExecutionReport
from src.utils.logging_config import setup_logging


@dataclass
class Position:
    region: str
    size: float
    average_price: float


class RiskManager:
    """Assess positions and enforce exposure limits."""

    def __init__(self, max_notional: float, correlation_limit: float) -> None:
        self.max_notional = max_notional
        self.correlation_limit = correlation_limit
        self.positions: Dict[str, Position] = {}
        self.logger = setup_logging(self.__class__.__name__)

    def update_positions(self, reports: Dict[str, ExecutionReport]) -> None:
        for key, report in reports.items():
            signal = report.signal
            position = self.positions.get(signal.region)
            size_change = signal.quantity if signal.action == "BUY" else -signal.quantity

            if position is None:
                self.positions[signal.region] = Position(signal.region, size_change, report.executed_price)
            else:
                new_size = position.size + size_change
                if new_size == 0:
                    del self.positions[signal.region]
                    continue
                new_avg_price = (
                    position.average_price * position.size + report.executed_price * size_change
                ) / new_size
                self.positions[signal.region] = Position(signal.region, new_size, new_avg_price)

    def check_exposure(self) -> bool:
        notional = sum(abs(pos.size * pos.average_price) for pos in self.positions.values())
        if notional > self.max_notional:
            self.logger.warning("Notional exposure %.2f exceeds limit %.2f", notional, self.max_notional)
            return False
        return True

    def check_correlation(self) -> bool:
        # Simplified correlation check based on region adjacency
        high_corr_regions = {
            "NSW1": {"QLD1", "VIC1"},
            "VIC1": {"NSW1", "SA1", "TAS1"},
            "QLD1": {"NSW1"},
            "SA1": {"VIC1"},
            "TAS1": {"VIC1"},
        }

        for region, position in self.positions.items():
            correlated_regions = high_corr_regions.get(region, set())
            correlated_notional = sum(
                abs(self.positions.get(r).size * self.positions.get(r).average_price)
                for r in correlated_regions
                if r in self.positions
            )
            own_notional = abs(position.size * position.average_price)
            if own_notional == 0:
                continue
            correlation_ratio = correlated_notional / own_notional
            if correlation_ratio > self.correlation_limit:
                self.logger.warning(
                    "Correlation exposure for %s exceeds limit (ratio=%.2f > %.2f)",
                    region,
                    correlation_ratio,
                    self.correlation_limit,
                )
                return False
        return True
