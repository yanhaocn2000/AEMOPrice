"""Run genetic optimisation across multiple strategy families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence

from src.backtesting.backtest_engine import BacktestEngine
from src.trading.analysis.strategy_comparator import FeatureRow, MarketRow
from src.trading.execution.order_executor import SimulatedOrderExecutor
from src.trading.optimization.genetic_optimizer import GeneticOptimizer, OptimizationResult, ParameterSpec
from src.trading.strategies.base import TradingStrategy
from src.utils.logging_config import setup_logging


@dataclass
class StrategyOptimizationConfig:
    """Configuration required to optimise a single strategy."""

    name: str
    parameter_specs: Sequence[ParameterSpec]
    strategy_builder: Callable[[Dict[str, float]], TradingStrategy]
    population_size: int | None = None
    generations: int | None = None
    elite_size: int | None = None
    random_seed: int | None = None


@dataclass
class StrategyOptimizationResult:
    name: str
    optimisation: OptimizationResult
    final_metrics: Dict[str, float]
    trade_count: int


@dataclass
class MultiStrategyReport:
    results: List[StrategyOptimizationResult]
    leader: StrategyOptimizationResult | None


class MultiStrategyOptimizer:
    """Coordinate the optimisation of multiple strategy families."""

    def __init__(
        self,
        region: str,
        initial_capital: float = 1_000_000.0,
        population_size: int = 14,
        generations: int = 12,
        elite_size: int = 2,
        random_seed: int | None = None,
    ) -> None:
        self.region = region
        self.initial_capital = initial_capital
        self.population_size = max(4, population_size)
        self.generations = max(1, generations)
        self.elite_size = max(1, elite_size)
        self.random_seed = random_seed
        self.logger = setup_logging(self.__class__.__name__)

    def optimise(
        self,
        market_data: Sequence[MarketRow],
        feature_data: Sequence[FeatureRow],
        configs: Sequence[StrategyOptimizationConfig],
    ) -> MultiStrategyReport:
        results: List[StrategyOptimizationResult] = []

        for index, config in enumerate(configs):
            seed = config.random_seed
            if seed is None and self.random_seed is not None:
                seed = self.random_seed + index

            optimizer = GeneticOptimizer(
                region=self.region,
                parameter_specs=config.parameter_specs,
                strategy_factory=config.strategy_builder,
                population_size=config.population_size or self.population_size,
                generations=config.generations or self.generations,
                elite_size=config.elite_size or self.elite_size,
                random_seed=seed,
            )

            self.logger.info("Optimising parameters for %s", config.name)
            optimisation_result = optimizer.optimise(market_data, feature_data)

            strategy = config.strategy_builder(dict(optimisation_result.best_params))
            executor = SimulatedOrderExecutor()
            engine = BacktestEngine(
                strategy=strategy,
                executor=executor,
                initial_capital=self.initial_capital,
            )
            backtest_result = engine.run(market_data, feature_data)

            results.append(
                StrategyOptimizationResult(
                    name=config.name,
                    optimisation=optimisation_result,
                    final_metrics=backtest_result.metrics,
                    trade_count=len(backtest_result.trades),
                )
            )

        leader = max(
            results,
            key=lambda item: item.final_metrics.get("total_return", 0.0),
            default=None,
        )
        return MultiStrategyReport(results=results, leader=leader)


def summarise_multi_report(report: MultiStrategyReport) -> List[Dict[str, float]]:
    """Flatten report results for easy rendering or persistence."""

    summary: List[Dict[str, float]] = []
    for result in report.results:
        item: Dict[str, float] = {
            "total_return": float(result.final_metrics.get("total_return", 0.0)),
            "sharpe": float(result.final_metrics.get("sharpe", 0.0)),
            "max_drawdown": float(result.final_metrics.get("max_drawdown", 0.0)),
            "trade_count": float(result.trade_count),
            "fitness": float(result.optimisation.fitness),
            "overfitting_penalty": float(result.optimisation.overfitting_penalty),
        }
        for key, value in result.optimisation.best_params.items():
            item[f"param_{key}"] = float(value)
        summary.append(item)
    return summary

