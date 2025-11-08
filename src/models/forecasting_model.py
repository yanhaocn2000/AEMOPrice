"""Simple forecasting model wrapper using lightweight heuristics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from src.utils.logging_config import setup_logging


Row = Dict[str, Any]


@dataclass
class ModelTrainingResult:
    model: "MeanReversionModel"
    metrics: Dict[str, float]


class MeanReversionModel:
    """Tiny model that predicts using historical averages."""

    def __init__(self) -> None:
        self.mean_target: float | None = None

    def fit(self, target: Iterable[float]) -> None:
        target_list = list(target)
        if not target_list:
            raise ValueError("Training target is empty")
        self.mean_target = sum(target_list) / len(target_list)

    def predict(self, count: int) -> List[float]:
        if self.mean_target is None:
            raise RuntimeError("Model has not been trained")
        return [self.mean_target for _ in range(count)]


class PriceForecaster:
    """Train and evaluate a baseline price forecasting model."""

    def __init__(self) -> None:
        self.logger = setup_logging(self.__class__.__name__)
        self.model = MeanReversionModel()

    def train(self, rows: List[Row], target_column: str) -> ModelTrainingResult:
        target = [float(row[target_column]) for row in rows]
        self.logger.info("Training MeanReversionModel on %d samples", len(target))
        self.model.fit(target)
        predictions = self.model.predict(len(target))
        metrics = self._calculate_metrics(target, predictions)
        return ModelTrainingResult(model=self.model, metrics=metrics)

    def predict(self, features: List[Row]) -> List[float]:
        return self.model.predict(len(features))

    def _calculate_metrics(self, actual: Iterable[float], predicted: Iterable[float]) -> Dict[str, float]:
        actual_list = list(actual)
        predicted_list = list(predicted)
        if not actual_list:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

        absolute_errors = [abs(a - p) for a, p in zip(actual_list, predicted_list)]
        mae = sum(absolute_errors) / len(absolute_errors)
        rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(actual_list, predicted_list)) / len(actual_list))

        mape_values = [abs((a - p) / a) for a, p in zip(actual_list, predicted_list) if a != 0]
        mape = (sum(mape_values) / len(mape_values) * 100) if mape_values else 0.0

        return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}
