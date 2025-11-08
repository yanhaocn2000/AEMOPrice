"""Simulated order execution module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.trading.strategies.base import TradeSignal
from src.utils.logging_config import setup_logging


@dataclass
class ExecutionReport:
    signal: TradeSignal
    executed_price: float
    slippage: float
    commission: float


class SimulatedOrderExecutor:
    """Execute trade signals with configurable transaction costs."""

    def __init__(self, commission_rate: float = 0.0005, slippage_rate: float = 0.001) -> None:
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.logger = setup_logging(self.__class__.__name__)

    def execute(self, signals: List[TradeSignal]) -> List[ExecutionReport]:
        reports: List[ExecutionReport] = []
        for signal in signals:
            slippage = signal.price * self.slippage_rate
            executed_price = signal.price + slippage if signal.action == "BUY" else signal.price - slippage
            commission = executed_price * abs(signal.quantity) * self.commission_rate

            report = ExecutionReport(signal, executed_price, slippage, commission)
            self.logger.debug(
                "Executed %s %s @ %.2f (slippage=%.4f, commission=%.4f)",
                signal.action,
                signal.region,
                executed_price,
                slippage,
                commission,
            )
            reports.append(report)
        return reports

