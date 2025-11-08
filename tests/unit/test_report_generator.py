import json
from pathlib import Path

from src.trading.analysis.report_generator import generate_ga_backtest_reports


def test_generate_ga_backtest_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    report = generate_ga_backtest_reports(
        output_dir,
        population_size=6,
        generations=4,
        elite_size=2,
        random_seed=3,
        initial_capital=600_000.0,
    )

    files = sorted(p.name for p in output_dir.iterdir())
    assert "summary.json" in files
    assert len(files) == 5

    summary_data = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_data["leader"] in {result.name for result in report.results}
    assert len(summary_data["results"]) == 4

    first_strategy_file = next(name for name in files if name != "summary.json")
    sample_report = json.loads((output_dir / first_strategy_file).read_text(encoding="utf-8"))
    assert "backtest" in sample_report
    assert "overfitting" in sample_report
    assert "is_overfitting" in sample_report["overfitting"]
