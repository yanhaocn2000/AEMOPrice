from datetime import datetime, timedelta
from typing import Dict

from datetime import datetime, timedelta

import pytest

from src.backtesting.backtest_engine import BacktestEngine
from src.trading.execution.order_executor import SimulatedOrderExecutor
from src.trading.optimization.genetic_optimizer import GeneticOptimizer, ParameterSpec
from src.trading.strategies.advanced_strategies import DualMovingAverageStrategy
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
    assert isinstance(result.is_overfitting, bool)
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


def test_genetic_optimizer_with_custom_parameters():
    market_data, feature_data = _build_dataset()

    def builder(params: Dict[str, float]) -> DualMovingAverageStrategy:
        return DualMovingAverageStrategy(
            region="NSW1",
            short_window=int(params["short_window"]),
            long_window=int(params["long_window"]),
            threshold=params["threshold"],
            base_lot=params["base_lot"],
            leverage=params["leverage"],
        )

    specs = [
        ParameterSpec("short_window", 3, 10, precision=1.0, is_integer=True),
        ParameterSpec("long_window", 12, 30, precision=1.0, is_integer=True),
        ParameterSpec("threshold", 0.0, 0.01, precision=0.0005),
        ParameterSpec("base_lot", 5.0, 60.0, precision=0.5),
        ParameterSpec("leverage", 2.0, 12.0, precision=0.5),
    ]

    optimizer = GeneticOptimizer(
        region="NSW1",
        parameter_specs=specs,
        strategy_factory=builder,
        population_size=6,
        generations=4,
        random_seed=3,
    )

    result = optimizer.optimise(market_data, feature_data)
    assert set(result.best_params.keys()) == {spec.name for spec in specs}
    assert len(result.history) == 4
    assert isinstance(result.is_overfitting, bool)

    strategy = builder(result.best_params)
    executor = SimulatedOrderExecutor()
    engine = BacktestEngine(strategy=strategy, executor=executor)
    backtest = engine.run(market_data, feature_data)

    assert "total_return" in backtest.metrics


