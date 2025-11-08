"""Genetic algorithm optimiser for trading strategy parameters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from src.backtesting.backtest_engine import BacktestEngine
from src.trading.execution.order_executor import SimulatedOrderExecutor
from src.trading.strategies.ml_strategy import MLTradingStrategy
from src.utils.logging_config import setup_logging


MarketRow = Mapping[str, Any]
FeatureRow = Mapping[str, Any]


@dataclass
class CandidateEvaluation:
    threshold: float
    lot_size: float
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
        threshold_range: Tuple[float, float],
        lot_size_range: Tuple[float, float],
        population_size: int = 12,
        generations: int = 15,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.2,
        train_ratio: float = 0.7,
        elite_size: int = 2,
        random_seed: int | None = None,
    ) -> None:
        import random

        self.region = region
        self.threshold_range = threshold_range
        self.lot_size_range = lot_size_range
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
        evaluations: Dict[Tuple[float, float], CandidateEvaluation] = {}
        history: List[Dict[str, float]] = []

        train_market, val_market, train_features, val_features = self._split_dataset(
            aligned_market, aligned_features
        )

        best_eval: CandidateEvaluation | None = None

        for generation in range(self.generations):
            generation_evals: List[CandidateEvaluation] = []
            for individual in population:
                key = (individual["threshold"], individual["lot_size"])
                cached = evaluations.get(key)
                if cached is None:
                    evaluation = self._evaluate_candidate(
                        individual, train_market, val_market, train_features, val_features
                    )
                    evaluations[key] = evaluation
                generation_evals.append(evaluations[key])

            generation_evals.sort(key=lambda item: item.fitness, reverse=True)
            best_eval = generation_evals[0] if generation_evals else best_eval
            if best_eval is None:
                raise RuntimeError("Failed to evaluate population")

            history.append({
                "generation": float(generation),
                "fitness": float(best_eval.fitness),
                "threshold": float(best_eval.threshold),
                "lot_size": float(best_eval.lot_size),
            })

            elites = [
                {"threshold": eval.threshold, "lot_size": eval.lot_size}
                for eval in generation_evals[: self.elite_size]
            ]

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
            best_params={"threshold": best_eval.threshold, "lot_size": best_eval.lot_size},
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
        boundaries = [
            {"threshold": self.threshold_range[0], "lot_size": self.lot_size_range[0]},
            {"threshold": self.threshold_range[1], "lot_size": self.lot_size_range[1]},
        ]

        for candidate in boundaries:
            population.append(self._clamp_and_round(candidate))

        while len(population) < self.population_size:
            threshold = self.random.uniform(*self.threshold_range)
            lot_size = self.random.uniform(*self.lot_size_range)
            population.append(
                self._clamp_and_round({"threshold": threshold, "lot_size": lot_size})
            )

        return population[: self.population_size]

    def _evaluate_candidate(
        self,
        candidate: Mapping[str, float],
        train_market: Sequence[MarketRow],
        val_market: Sequence[MarketRow],
        train_features: Sequence[FeatureRow],
        val_features: Sequence[FeatureRow],
    ) -> CandidateEvaluation:
        threshold = candidate["threshold"]
        lot_size = candidate["lot_size"]

        strategy = MLTradingStrategy(region=self.region, threshold=threshold, lot_size=lot_size)
        executor = SimulatedOrderExecutor()

        train_engine = BacktestEngine(strategy=strategy, executor=executor)
        train_result = train_engine.run(train_market, train_features)

        # Re-initialise strategy/executor for validation to avoid state leakage
        val_strategy = MLTradingStrategy(region=self.region, threshold=threshold, lot_size=lot_size)
        val_executor = SimulatedOrderExecutor()
        val_engine = BacktestEngine(strategy=val_strategy, executor=val_executor)
        val_result = val_engine.run(val_market, val_features)

        train_return = train_result.metrics.get("total_return", 0.0)
        val_return = val_result.metrics.get("total_return", 0.0)
        penalty = abs(train_return - val_return)
        fitness = 0.6 * train_return + 0.4 * val_return - penalty

        return CandidateEvaluation(
            threshold=float(threshold),
            lot_size=float(lot_size),
            fitness=float(fitness),
            train_metrics=train_result.metrics,
            validation_metrics=val_result.metrics,
        )

    def _tournament_select(self, evaluations: Sequence[CandidateEvaluation]) -> Dict[str, float]:
        k = min(3, len(evaluations))
        contenders = self.random.sample(list(evaluations), k=k)
        winner = max(contenders, key=lambda item: item.fitness)
        return {"threshold": winner.threshold, "lot_size": winner.lot_size}

    def _crossover(self, parent_a: Mapping[str, float], parent_b: Mapping[str, float]) -> Dict[str, float]:
        if self.random.random() >= self.crossover_rate:
            return self._clamp_and_round(dict(parent_a))

        threshold = (parent_a["threshold"] + parent_b["threshold"]) / 2
        lot_size = (parent_a["lot_size"] + parent_b["lot_size"]) / 2
        return self._clamp_and_round({"threshold": threshold, "lot_size": lot_size})

    def _mutate(self, individual: Mapping[str, float]) -> Dict[str, float]:
        mutated = dict(individual)

        if self.random.random() < self.mutation_rate:
            span = self.threshold_range[1] - self.threshold_range[0]
            mutated["threshold"] += (self.random.random() - 0.5) * span * 0.2

        if self.random.random() < self.mutation_rate:
            span = self.lot_size_range[1] - self.lot_size_range[0]
            mutated["lot_size"] += (self.random.random() - 0.5) * span * 0.2

        return self._clamp_and_round(mutated)

    def _clamp_and_round(self, values: Mapping[str, float]) -> Dict[str, float]:
        threshold = min(max(values["threshold"], self.threshold_range[0]), self.threshold_range[1])
        lot_size = min(max(values["lot_size"], self.lot_size_range[0]), self.lot_size_range[1])
        return {"threshold": round(threshold, 4), "lot_size": round(lot_size, 4)}

    def _ensure_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))

