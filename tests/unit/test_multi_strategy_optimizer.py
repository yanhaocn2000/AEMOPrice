from typing import Dict

from src.trading.analysis.datasets import multi_strategy_benchmark_dataset
from src.trading.analysis.multi_strategy_optimizer import (
    MultiStrategyOptimizer,
    StrategyOptimizationConfig,
    summarise_multi_report,
)
from src.trading.optimization.genetic_optimizer import ParameterSpec
from src.trading.strategies.advanced_strategies import (
    CarryMomentumStrategy,
    DualMovingAverageStrategy,
    RegimeSwitchingStrategy,
)


def test_multi_strategy_optimizer_produces_leader():
    market_data, feature_data = multi_strategy_benchmark_dataset()

    def build_dual_ma(params: Dict[str, float]) -> DualMovingAverageStrategy:
        short = max(3, int(round(params["short_window"])))
        long = max(short + 1, int(round(params["long_window"])))
        return DualMovingAverageStrategy(
            region="NSW1",
            short_window=short,
            long_window=long,
            threshold=params["threshold"],
            base_lot=params["base_lot"],
            leverage=params["leverage"],
        )

    def build_carry(params: Dict[str, float]) -> CarryMomentumStrategy:
        return CarryMomentumStrategy(
            region="NSW1",
            carry_threshold=params["carry_threshold"],
            momentum_weight=params["momentum_weight"],
            base_lot=params["base_lot"],
            risk_multiplier=params["risk_multiplier"],
            max_lot=params["max_lot"],
        )

    def build_regime(params: Dict[str, float]) -> RegimeSwitchingStrategy:
        window = max(5, int(round(params["volatility_window"])))
        return RegimeSwitchingStrategy(
            region="NSW1",
            volatility_window=window,
            calm_threshold=params["calm_threshold"],
            trend_threshold=params["trend_threshold"],
            reversion_threshold=params["reversion_threshold"],
            base_lot=params["base_lot"],
            max_lot=params["max_lot"],
        )

    configs = [
        StrategyOptimizationConfig(
            name="DualMovingAverage",
            parameter_specs=[
                ParameterSpec("short_window", 4, 12, precision=1.0, is_integer=True),
                ParameterSpec("long_window", 14, 36, precision=1.0, is_integer=True),
                ParameterSpec("threshold", 0.0, 0.01, precision=0.0005),
                ParameterSpec("base_lot", 10.0, 120.0, precision=1.0),
                ParameterSpec("leverage", 4.0, 16.0, precision=0.5),
            ],
            strategy_builder=build_dual_ma,
            population_size=6,
            generations=5,
        ),
        StrategyOptimizationConfig(
            name="CarryMomentum",
            parameter_specs=[
                ParameterSpec("carry_threshold", 0.0, 0.01, precision=0.0005),
                ParameterSpec("momentum_weight", 1.0, 5.0, precision=0.5),
                ParameterSpec("base_lot", 15.0, 150.0, precision=1.0),
                ParameterSpec("risk_multiplier", 200.0, 1200.0, precision=10.0),
                ParameterSpec("max_lot", 150.0, 900.0, precision=10.0),
            ],
            strategy_builder=build_carry,
            population_size=6,
            generations=5,
        ),
        StrategyOptimizationConfig(
            name="RegimeSwitching",
            parameter_specs=[
                ParameterSpec("volatility_window", 6, 20, precision=1.0, is_integer=True),
                ParameterSpec("calm_threshold", 0.002, 0.02, precision=0.0005),
                ParameterSpec("trend_threshold", 0.0005, 0.01, precision=0.0005),
                ParameterSpec("reversion_threshold", 0.0005, 0.01, precision=0.0005),
                ParameterSpec("base_lot", 20.0, 200.0, precision=1.0),
                ParameterSpec("max_lot", 200.0, 1000.0, precision=10.0),
            ],
            strategy_builder=build_regime,
            population_size=6,
            generations=5,
        ),
    ]

    optimizer = MultiStrategyOptimizer(
        region="NSW1",
        initial_capital=500_000.0,
        population_size=6,
        generations=5,
        random_seed=5,
    )

    report = optimizer.optimise(market_data, feature_data, configs)

    assert len(report.results) == len(configs)
    assert report.leader in report.results
    for result in report.results:
        assert result.final_metrics["total_return"] > -1.0
        assert result.trade_count >= 0
        assert len(result.optimisation.history) == 5
        assert isinstance(result.optimisation.is_overfitting, bool)
        assert isinstance(result.final_capital, float)
        assert isinstance(result.equity_curve, list)
        assert isinstance(result.trades, list)

    summary = summarise_multi_report(report)
    assert len(summary) == len(configs)
    assert any(item["total_return"] > 0 for item in summary)

