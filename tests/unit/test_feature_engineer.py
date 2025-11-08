from src.features.feature_engineer import FeatureConfig, FeatureEngineer


def test_feature_engineer_generates_lags_and_rolling():
    rows = [
        {"timestamp": f"2024-01-01T0{i}:00:00", "price": 100 + i}
        for i in range(5)
    ]
    config = FeatureConfig(target_column="price", lag_periods=[1], rolling_windows=[2])
    engineer = FeatureEngineer(config)
    result = engineer.transform(rows)

    assert result[0]["price_lag_1"] == 100.0
    assert result[0]["price_rolling_mean_2"] == 100.5
    assert "target_future_return" in result[0]
    assert all(value is not None for row in result for value in row.values())
