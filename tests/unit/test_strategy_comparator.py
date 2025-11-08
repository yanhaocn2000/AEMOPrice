from datetime import datetime, timedelta

from src.backtesting.backtest_engine import BacktestEngine
from src.trading.analysis.strategy_comparator import StrategyComparator, summarise_report
from src.trading.execution.order_executor import SimulatedOrderExecutor
from src.trading.strategies.ml_strategy import MLTradingStrategy
from src.trading.strategies.rule_based_strategies import (
    AdaptiveThresholdStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    VolatilityBreakoutStrategy,
)


def _build_dataset(length: int = 48):
    start = datetime(2024, 1, 1)
    market_data = []
    feature_data = []
    price = 100.0
    for idx in range(length):
        timestamp = start + timedelta(minutes=5 * idx)
        cycle = idx % 6
        drift = 0.01 if cycle in (0, 1) else -0.006 if cycle in (2, 3) else 0.015
        price *= 1 + drift
        predicted_return = drift * 0.8
        market_data.append({"timestamp": timestamp.isoformat(), "price": price, "region": "NSW1"})
        feature_data.append(
            {
                "timestamp": timestamp.isoformat(),
                "predicted_return": predicted_return,
                "target_future_return": drift,
            }
        )
    return market_data, feature_data


def test_rule_based_strategies_generate_trades():
    market_data, feature_data = _build_dataset()

    strategies = [
        MLTradingStrategy(region="NSW1", threshold=0.003, lot_size=5.0),
        MeanReversionStrategy(region="NSW1", lookback=6, entry_zscore=0.5, lot_size=4.0),
        MomentumStrategy(region="NSW1", window=4, return_threshold=0.001, lot_size=4.0),
        VolatilityBreakoutStrategy(region="NSW1", lookback=6, volatility_multiplier=1.2, lot_size=4.0),
        AdaptiveThresholdStrategy(region="NSW1", base_threshold=0.001, risk_aversion=2500.0, max_lot=12.0),
    ]

    comparator = StrategyComparator(initial_capital=500_000.0)
    report = comparator.compare(market_data, feature_data, strategies)
    summary = summarise_report(report)

    assert len(report.performances) == len(strategies)
    assert len(summary) == len(strategies)
    assert report.leader is not None
    for item in summary:
        name, total_return, sharpe, max_dd, trade_count = item
        assert isinstance(name, str)
        assert trade_count >= 0
        assert isinstance(total_return, float)
        assert isinstance(sharpe, float)
        assert isinstance(max_dd, float)

    # Ensure the ML baseline still functions as before
    engine = BacktestEngine(
        strategy=MLTradingStrategy(region="NSW1", threshold=0.003, lot_size=5.0),
        executor=SimulatedOrderExecutor(),
    )
    result = engine.run(market_data, feature_data)
    assert len(result.trades) > 0
    assert "total_return" in result.metrics

