# GA 回测报告说明

本目录存放使用遗传算法调参与回测得到的详细 JSON 报告，包含以下内容：

- `*_report.json`：对应策略（DualMovingAverage、CarryMomentum、RegimeSwitching、AdaptiveThreshold）在训练、验证与全样本集上的收益、风险指标、交易记录与权益曲线。
- `summary.json`：综合统计各策略的最优参数、收益表现与过拟合判定结果。

## 使用方式

1. 运行 `src/trading/analysis/report_generator.py` 中的 `generate_ga_backtest_reports` 函数即可重新生成全部报告。
2. 生成过程中会自动调用遗传算法优化器，并使用训练/验证收益差异计算过拟合惩罚并写入 `is_overfitting` 标志。
3. 如需新增策略，只需实现新的 `TradingStrategy` 子类，并在报告生成配置中注册该策略及其参数搜索空间。

## 注意事项

- 当前示例数据集位于 `src/trading/analysis/datasets.py` 中，为合成的高增长场景，实际应用时需替换为真实的市场数据。
- 若在离线环境中运行，请确认 `reports/ga_backtests/` 目录存在写入权限以确保报告可以正确输出。

