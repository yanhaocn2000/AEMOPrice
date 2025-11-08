from datetime import datetime, timedelta

import pytest

from src.backtesting.backtest_engine import BacktestEngine
from src.trading.execution.order_executor import SimulatedOrderExecutor
from src.trading.optimization.genetic_optimizer import GeneticOptimizer
from src.trading.strategies.ml_strategy import MLTradingStrategy


def _build_dataset(length: int = 60):
    start = datetime(2024, 1, 1, 0, 0, 0)
    market_data = []
    feature_data = []
    price = 100.0

    for idx in range(length):
        timestamp = start + timedelta(minutes=5 * idx)
        # Generate a smooth oscillating return profile
        actual_return = 0.01 if idx % 4 in (0, 1) else -0.008
        predicted_return = actual_return * 0.8

        price *= 1 + actual_return
        market_data.append({"timestamp": timestamp.isoformat(), "price": price, "region": "NSW1"})
        feature_data.append(
            {
                "timestamp": timestamp.isoformat(),
                "predicted_return": predicted_return,
                "target_future_return": actual_return,
            }
        )

    return market_data, feature_data


def test_genetic_optimizer_returns_reasonable_parameters():
    market_data, feature_data = _build_dataset()
    optimizer = GeneticOptimizer(
        region="NSW1",
        threshold_range=(0.0, 0.05),
        lot_size_range=(1.0, 5.0),
        population_size=8,
        generations=6,
        random_seed=7,
    )

    result = optimizer.optimise(market_data, feature_data)

    assert 0.0 <= result.best_params["threshold"] <= 0.05
    assert 1.0 <= result.best_params["lot_size"] <= 5.0
    assert result.overfitting_penalty == pytest.approx(
        abs(result.train_metrics["total_return"] - result.validation_metrics["total_return"])
    )
    assert len(result.history) == 6

    # Run a final backtest with the optimised parameters to ensure trades are produced
    strategy = MLTradingStrategy(
        region="NSW1",
        threshold=result.best_params["threshold"],
        lot_size=result.best_params["lot_size"],
    )
    executor = SimulatedOrderExecutor()
    engine = BacktestEngine(strategy=strategy, executor=executor)
    final_result = engine.run(market_data, feature_data)

    assert len(final_result.trades) > 0
    assert "total_return" in final_result.metrics

