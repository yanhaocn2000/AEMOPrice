"""Utilities to generate and persist GA-tuned backtest reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from src.trading.analysis.datasets import high_growth_dataset
from src.trading.analysis.multi_strategy_optimizer import (
    MultiStrategyOptimizer,
    MultiStrategyReport,
    StrategyOptimizationConfig,
)
from src.trading.optimization.genetic_optimizer import ParameterSpec
from src.trading.strategies.advanced_strategies import (
    CarryMomentumStrategy,
    DualMovingAverageStrategy,
    RegimeSwitchingStrategy,
)
from src.trading.strategies.rule_based_strategies import AdaptiveThresholdStrategy


def _build_dual_ma(params: Dict[str, float]) -> DualMovingAverageStrategy:
    short = max(3, int(round(params["short_window"])))
    long = max(short + 1, int(round(params["long_window"])))
    return DualMovingAverageStrategy(
        region="NSW1",
        short_window=short,
        long_window=long,
        threshold=params["threshold"],
        base_lot=params["base_lot"],
        leverage=params["leverage"],
        max_lot=params["max_lot"],
    )


def _build_carry(params: Dict[str, float]) -> CarryMomentumStrategy:
    return CarryMomentumStrategy(
        region="NSW1",
        carry_threshold=params["carry_threshold"],
        momentum_weight=params["momentum_weight"],
        base_lot=params["base_lot"],
        risk_multiplier=params["risk_multiplier"],
        max_lot=params["max_lot"],
    )


def _build_regime(params: Dict[str, float]) -> RegimeSwitchingStrategy:
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


def _build_adaptive(params: Dict[str, float]) -> AdaptiveThresholdStrategy:
    return AdaptiveThresholdStrategy(
        region="NSW1",
        base_threshold=params["base_threshold"],
        risk_aversion=params["risk_aversion"],
        max_lot=params["max_lot"],
    )


def _strategy_configs() -> Sequence[StrategyOptimizationConfig]:
    return (
        StrategyOptimizationConfig(
            name="DualMovingAverage",
            parameter_specs=[
                ParameterSpec("short_window", 4, 18, precision=1.0, is_integer=True),
                ParameterSpec("long_window", 20, 64, precision=1.0, is_integer=True),
                ParameterSpec("threshold", 0.0, 0.02, precision=0.0005),
                ParameterSpec("base_lot", 15.0, 160.0, precision=1.0),
                ParameterSpec("leverage", 4.0, 18.0, precision=0.5),
                ParameterSpec("max_lot", 200.0, 1200.0, precision=10.0),
            ],
            strategy_builder=_build_dual_ma,
        ),
        StrategyOptimizationConfig(
            name="CarryMomentum",
            parameter_specs=[
                ParameterSpec("carry_threshold", 0.0, 0.02, precision=0.0005),
                ParameterSpec("momentum_weight", 0.5, 6.0, precision=0.5),
                ParameterSpec("base_lot", 20.0, 200.0, precision=1.0),
                ParameterSpec("risk_multiplier", 200.0, 1600.0, precision=10.0),
                ParameterSpec("max_lot", 200.0, 1200.0, precision=10.0),
            ],
            strategy_builder=_build_carry,
        ),
        StrategyOptimizationConfig(
            name="RegimeSwitching",
            parameter_specs=[
                ParameterSpec("volatility_window", 6, 24, precision=1.0, is_integer=True),
                ParameterSpec("calm_threshold", 0.001, 0.02, precision=0.0005),
                ParameterSpec("trend_threshold", 0.0005, 0.015, precision=0.0005),
                ParameterSpec("reversion_threshold", 0.0005, 0.015, precision=0.0005),
                ParameterSpec("base_lot", 20.0, 220.0, precision=1.0),
                ParameterSpec("max_lot", 200.0, 1400.0, precision=10.0),
            ],
            strategy_builder=_build_regime,
        ),
        StrategyOptimizationConfig(
            name="AdaptiveThreshold",
            parameter_specs=[
                ParameterSpec("base_threshold", 0.0005, 0.01, precision=0.0005),
                ParameterSpec("risk_aversion", 500.0, 6000.0, precision=50.0),
                ParameterSpec("max_lot", 100.0, 1500.0, precision=10.0),
            ],
            strategy_builder=_build_adaptive,
        ),
    )


def _serialise_result(result) -> Dict[str, object]:
    return {
        "strategy": result.name,
        "best_params": dict(result.optimisation.best_params),
        "train_metrics": result.optimisation.train_metrics,
        "validation_metrics": result.optimisation.validation_metrics,
        "overfitting": {
            "penalty": float(result.optimisation.overfitting_penalty),
            "is_overfitting": bool(result.optimisation.is_overfitting),
        },
        "fitness": float(result.optimisation.fitness),
        "history": result.optimisation.history,
        "backtest": {
            "metrics": result.final_metrics,
            "trade_count": int(result.trade_count),
            "final_capital": float(result.final_capital),
            "equity_curve": result.equity_curve,
            "trades": result.trades,
        },
    }


def _write_report(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def generate_ga_backtest_reports(
    output_dir: str | Path,
    *,
    population_size: int = 12,
    generations: int = 10,
    elite_size: int = 2,
    random_seed: int = 11,
    initial_capital: float = 1_000_000.0,
) -> MultiStrategyReport:
    """Optimise four strategy families and persist detailed reports."""

    output_path = Path(output_dir)
    market_data, feature_data = high_growth_dataset(length=240)

    optimizer = MultiStrategyOptimizer(
        region="NSW1",
        initial_capital=initial_capital,
        population_size=population_size,
        generations=generations,
        elite_size=elite_size,
        random_seed=random_seed,
    )

    configs = _strategy_configs()
    report = optimizer.optimise(market_data, feature_data, configs)

    summary: List[Dict[str, object]] = []
    for result in report.results:
        payload = _serialise_result(result)
        _write_report(output_path / f"{result.name.lower()}_report.json", payload)
        summary.append(payload)

    leader_name = report.leader.name if report.leader else None
    _write_report(
        output_path / "summary.json",
        {
            "leader": leader_name,
            "results": summary,
        },
    )

    return report


__all__ = ["generate_ga_backtest_reports"]

