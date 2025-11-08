"""Genetic algorithm optimiser for trading strategy parameters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from src.backtesting.backtest_engine import BacktestEngine
from src.trading.execution.order_executor import SimulatedOrderExecutor
from src.trading.strategies.base import TradingStrategy
from src.trading.strategies.ml_strategy import MLTradingStrategy
from src.utils.logging_config import setup_logging


MarketRow = Mapping[str, Any]
FeatureRow = Mapping[str, Any]


@dataclass
class ParameterSpec:
    """Describe the bounds and rounding of an optimisable parameter."""

    name: str
    lower: float
    upper: float
    precision: float = 0.0001
    is_integer: bool = False


@dataclass
class CandidateEvaluation:
    params: Dict[str, float]
    fitness: float
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]


@dataclass
class OptimizationResult:
    best_params: Dict[str, float]
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    overfitting_penalty: float
    fitness: float
    history: List[Dict[str, float]]


class GeneticOptimizer:
    """Optimise strategy parameters using a simple genetic algorithm."""

    def __init__(
        self,
        region: str,
        threshold_range: Tuple[float, float] | None = None,
        lot_size_range: Tuple[float, float] | None = None,
        *,
        parameter_specs: Sequence[ParameterSpec] | None = None,
        strategy_factory: Callable[[Dict[str, float]], TradingStrategy] | None = None,
        population_size: int = 12,
        generations: int = 15,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.2,
        train_ratio: float = 0.7,
        elite_size: int = 2,
        random_seed: int | None = None,
    ) -> None:
        import random

        if parameter_specs is None:
            if threshold_range is None or lot_size_range is None:
                raise ValueError(
                    "threshold_range and lot_size_range must be provided when parameter_specs are omitted"
                )
            parameter_specs = (
                ParameterSpec("threshold", threshold_range[0], threshold_range[1]),
                ParameterSpec("lot_size", lot_size_range[0], lot_size_range[1]),
            )
        else:
            if threshold_range is not None or lot_size_range is not None:
                raise ValueError(
                    "Do not provide threshold_range/lot_size_range when parameter_specs are supplied"
                )

        self.region = region
        self.parameter_specs = list(parameter_specs)
        if not self.parameter_specs:
            raise ValueError("At least one parameter specification must be provided")
        self.parameter_names = [spec.name for spec in self.parameter_specs]

        if strategy_factory is None:
            def default_factory(params: Dict[str, float]) -> TradingStrategy:
                return MLTradingStrategy(
                    region=region,
                    threshold=float(params.get("threshold", 0.0)),
                    lot_size=float(params.get("lot_size", 1.0)),
                )

            self.strategy_factory = default_factory
        else:
            self.strategy_factory = strategy_factory

        self.population_size = max(4, population_size)
        self.generations = max(1, generations)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.train_ratio = min(0.9, max(0.1, train_ratio))
        self.elite_size = max(1, min(elite_size, self.population_size - 1))
        self.random = random.Random(random_seed)
        self.logger = setup_logging(self.__class__.__name__)

    def optimise(
        self,
        market_data: Sequence[MarketRow],
        feature_data: Sequence[FeatureRow],
    ) -> OptimizationResult:
        self.logger.info(
            "Starting GA optimisation (population=%d, generations=%d)",
            self.population_size,
            self.generations,
        )
        aligned_market, aligned_features = self._align_data(market_data, feature_data)
        if len(aligned_market) < 4:
            raise ValueError("Insufficient aligned data for optimisation")

        population = self._initialise_population()
        evaluations: Dict[Tuple[float, ...], CandidateEvaluation] = {}
        history: List[Dict[str, float]] = []

        train_market, val_market, train_features, val_features = self._split_dataset(
            aligned_market, aligned_features
        )

        best_eval: CandidateEvaluation | None = None

        for generation in range(self.generations):
            generation_evals: List[CandidateEvaluation] = []
            for individual in population:
                key = self._make_key(individual)
                cached = evaluations.get(key)
                if cached is None:
                    evaluation = self._evaluate_candidate(
                        individual, train_market, val_market, train_features, val_features
                    )
                    evaluations[key] = evaluation
                    cached = evaluation
                generation_evals.append(cached)

            generation_evals.sort(key=lambda item: item.fitness, reverse=True)
            best_eval = generation_evals[0] if generation_evals else best_eval
            if best_eval is None:
                raise RuntimeError("Failed to evaluate population")

            snapshot = {"generation": float(generation), "fitness": float(best_eval.fitness)}
            for name in self.parameter_names:
                snapshot[name] = float(best_eval.params[name])
            history.append(snapshot)

            elites = [dict(eval.params) for eval in generation_evals[: self.elite_size]]

            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent_a = self._tournament_select(generation_evals)
                parent_b = self._tournament_select(generation_evals)
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

        if best_eval is None:
            raise RuntimeError("Optimisation did not produce a best candidate")

        penalty = abs(
            best_eval.train_metrics.get("total_return", 0.0)
            - best_eval.validation_metrics.get("total_return", 0.0)
        )

        return OptimizationResult(
            best_params=dict(best_eval.params),
            train_metrics=best_eval.train_metrics,
            validation_metrics=best_eval.validation_metrics,
            overfitting_penalty=penalty,
            fitness=best_eval.fitness,
            history=history,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _align_data(
        self,
        market_data: Sequence[MarketRow],
        feature_data: Sequence[FeatureRow],
    ) -> Tuple[List[MarketRow], List[FeatureRow]]:
        feature_lookup = {
            self._ensure_datetime(row["timestamp"]): row for row in feature_data if "timestamp" in row
        }

        aligned_market: List[MarketRow] = []
        aligned_features: List[FeatureRow] = []

        for market_row in sorted(market_data, key=lambda row: self._ensure_datetime(row["timestamp"])):
            timestamp = self._ensure_datetime(market_row["timestamp"])
            feature_row = feature_lookup.get(timestamp)
            if feature_row is None:
                continue
            aligned_market.append(market_row)
            aligned_features.append(feature_row)

        return aligned_market, aligned_features

    def _split_dataset(
        self,
        market_data: Sequence[MarketRow],
        feature_data: Sequence[FeatureRow],
    ) -> Tuple[List[MarketRow], List[MarketRow], List[FeatureRow], List[FeatureRow]]:
        split_index = int(len(market_data) * self.train_ratio)
        split_index = max(1, min(split_index, len(market_data) - 1))

        train_market = list(market_data[:split_index])
        val_market = list(market_data[split_index:]) or list(market_data[-1:])
        train_features = list(feature_data[:split_index])
        val_features = list(feature_data[split_index:]) or list(feature_data[-1:])

        return train_market, val_market, train_features, val_features

    def _initialise_population(self) -> List[Dict[str, float]]:
        population: List[Dict[str, float]] = []

        lower_candidate = {spec.name: spec.lower for spec in self.parameter_specs}
        upper_candidate = {spec.name: spec.upper for spec in self.parameter_specs}
        population.append(self._clamp_and_round(lower_candidate))
        population.append(self._clamp_and_round(upper_candidate))

        while len(population) < self.population_size:
            candidate = {
                spec.name: self.random.uniform(spec.lower, spec.upper)
                for spec in self.parameter_specs
            }
            population.append(self._clamp_and_round(candidate))

        return population[: self.population_size]

    def _evaluate_candidate(
        self,
        candidate: Mapping[str, float],
        train_market: Sequence[MarketRow],
        val_market: Sequence[MarketRow],
        train_features: Sequence[FeatureRow],
        val_features: Sequence[FeatureRow],
    ) -> CandidateEvaluation:
        params = self._clamp_and_round(candidate)

        strategy_train = self.strategy_factory(dict(params))
        train_engine = BacktestEngine(strategy=strategy_train, executor=SimulatedOrderExecutor())
        train_result = train_engine.run(train_market, train_features)

        strategy_val = self.strategy_factory(dict(params))
        val_engine = BacktestEngine(strategy=strategy_val, executor=SimulatedOrderExecutor())
        val_result = val_engine.run(val_market, val_features)

        train_return = float(train_result.metrics.get("total_return", 0.0))
        val_return = float(val_result.metrics.get("total_return", 0.0))
        penalty = abs(train_return - val_return)
        fitness = 0.6 * train_return + 0.4 * val_return - penalty

        return CandidateEvaluation(
            params=params,
            fitness=float(fitness),
            train_metrics=train_result.metrics,
            validation_metrics=val_result.metrics,
        )

    def _tournament_select(self, evaluations: Sequence[CandidateEvaluation]) -> Dict[str, float]:
        k = min(3, len(evaluations))
        contenders = self.random.sample(list(evaluations), k=k)
        winner = max(contenders, key=lambda item: item.fitness)
        return dict(winner.params)

    def _crossover(self, parent_a: Mapping[str, float], parent_b: Mapping[str, float]) -> Dict[str, float]:
        if self.random.random() >= self.crossover_rate:
            return self._clamp_and_round(parent_a)

        child: Dict[str, float] = {}
        for spec in self.parameter_specs:
            value = (parent_a[spec.name] + parent_b[spec.name]) / 2
            child[spec.name] = value
        return self._clamp_and_round(child)

    def _mutate(self, individual: Mapping[str, float]) -> Dict[str, float]:
        mutated = dict(individual)

        for spec in self.parameter_specs:
            if self.random.random() < self.mutation_rate:
                span = spec.upper - spec.lower
                if span <= 0:
                    continue
                mutated[spec.name] += (self.random.random() - 0.5) * span * 0.2

        return self._clamp_and_round(mutated)

    def _clamp_and_round(self, values: Mapping[str, float]) -> Dict[str, float]:
        clamped: Dict[str, float] = {}
        for spec in self.parameter_specs:
            value = values.get(spec.name, spec.lower)
            value = min(max(value, spec.lower), spec.upper)
            if spec.is_integer:
                value = float(int(round(value)))
            else:
                precision = spec.precision if spec.precision > 0 else 0.0001
                value = round(value / precision) * precision
            value = min(max(value, spec.lower), spec.upper)
            clamped[spec.name] = float(value)
        return clamped

    def _make_key(self, values: Mapping[str, float]) -> Tuple[float, ...]:
        return tuple(float(values[name]) for name in self.parameter_names)

    def _ensure_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))
