"""Common strategy abstractions and signal definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

from src.utils.logging_config import setup_logging


@dataclass
class TradeSignal:
    """Normalized representation of a trade decision."""

    timestamp: datetime
    region: str
    action: str
    quantity: float
    price: float


class TradingStrategy(ABC):
    """Abstract base class implemented by all strategies."""

    def __init__(self, region: str) -> None:
        self.region = region
        self.logger = setup_logging(self.__class__.__name__)

    @abstractmethod
    def generate_signals(
        self,
        market_row: Mapping[str, Any],
        feature_row: Mapping[str, Any],
    ) -> Sequence[TradeSignal]:
        """Return trade signals for the provided market snapshot."""

