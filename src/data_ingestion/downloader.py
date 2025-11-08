"""Data downloading utilities for the NEM trading system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from src.utils.logging_config import setup_logging


Row = Dict[str, Any]


class AEMODataDownloader:
    """Download AEMO market data and cache the results locally."""

    def __init__(
        self,
        base_url: str,
        cache_dir: Path,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.session = session or requests.Session()
        self.logger = setup_logging(self.__class__.__name__)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_market_data(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> List[Row]:
        """Download market data from the configured AEMO endpoint."""

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        self.logger.info("Downloading data from %s", url)
        response = self.session.get(url, params=params, timeout=60)
        response.raise_for_status()

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            self.logger.error("Failed to decode response from %s", url)
            raise RuntimeError("Invalid JSON payload") from exc

        if isinstance(payload, dict):
            payload = [payload]

        if not payload:
            self.logger.warning("Received empty dataset from %s", url)
        return list(payload)

    def save_to_parquet(self, rows: Iterable[Row], filename: str) -> Path:
        """Persist records to the cache directory using JSON storage."""

        target_path = self.cache_dir / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as handle:
            json.dump(list(rows), handle)
        self.logger.info("Saved dataset to %s", target_path)
        return target_path

    def bulk_download(self, requests_config: Iterable[Dict[str, Dict[str, str]]]) -> Dict[str, List[Row]]:
        """Download a batch of datasets defined by ``requests_config``."""

        results: Dict[str, List[Row]] = {}
        for item in requests_config:
            endpoint = item["endpoint"]
            params = item.get("params")
            key = item.get("name", endpoint)
            rows = self.download_market_data(endpoint, params)
            results[key] = rows
        return results
