from datetime import datetime

from src.risk.risk_manager import RiskManager
from src.trading.execution.order_executor import ExecutionReport
from src.trading.strategies.base import TradeSignal


def make_report(region: str, action: str, price: float, quantity: float) -> ExecutionReport:
    signal = TradeSignal(
        timestamp=datetime(2024, 1, 1),
        region=region,
        action=action,
        quantity=quantity,
        price=price,
    )
    return ExecutionReport(signal=signal, executed_price=price, slippage=0.1, commission=1.0)


def test_risk_manager_exposure_limit():
    manager = RiskManager(max_notional=1_000, correlation_limit=2.0)
    reports = {
        "trade1": make_report("NSW1", "BUY", 100.0, 5.0),
        "trade2": make_report("VIC1", "BUY", 100.0, 6.0),
    }
    manager.update_positions(reports)
    assert not manager.check_exposure()


def test_risk_manager_correlation_limit():
    manager = RiskManager(max_notional=10_000, correlation_limit=0.5)
    reports = {
        "trade1": make_report("NSW1", "BUY", 100.0, 1.0),
        "trade2": make_report("QLD1", "BUY", 100.0, 1.0),
    }
    manager.update_positions(reports)
    assert not manager.check_correlation()

