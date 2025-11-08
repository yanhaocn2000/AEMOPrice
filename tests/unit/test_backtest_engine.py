from src.backtesting.backtest_engine import BacktestEngine
from src.trading.execution.order_executor import SimulatedOrderExecutor
from src.trading.strategies.ml_strategy import MLTradingStrategy


def test_backtest_engine_runs_basic_loop():
    timestamps = [f"2024-01-01T{str(i).zfill(2)}:00:00" for i in range(10)]
    market_data = [
        {"timestamp": ts, "price": 100 + idx, "region": "NSW1"}
        for idx, ts in enumerate(timestamps)
    ]

    feature_data = []
    for ts in timestamps:
        feature_data.append(
            {
                "timestamp": ts,
                "predicted_return": 0.02,
                "target_future_return": 0.015,
            }
        )

    strategy = MLTradingStrategy(region="NSW1", threshold=0.01, lot_size=1)
    executor = SimulatedOrderExecutor()
    engine = BacktestEngine(strategy=strategy, executor=executor)

    result = engine.run(market_data, feature_data)

    assert result.equity_curve[0].value >= engine.initial_capital - 1
    assert len(result.trades) > 0
    assert "total_return" in result.metrics
