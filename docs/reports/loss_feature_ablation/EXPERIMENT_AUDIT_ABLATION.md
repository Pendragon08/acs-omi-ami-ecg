# 损失函数/特征消融完整性审计（历史 OMI/AMI 任务）

> 本审计对应旧的 OMI/AMI 二分类消融，不是当前 ACS/OMI 联合任务的评价。

> 中文说明：本文件审计固定划分、单变量改变、训练集权重统计和测试集隔离；
> 外部测试标签未参与调参。

Overall: **WARN** (provisional deterministic audit; no fresh independent semantic reviewer was authorized).

- Parsed and recomputed 63 run JSON files and their validation prediction CSVs.
- Baseline checkpoint exactly reproduces the historical checkpoint: True.
- Ablation runner references the external test: False.
- Final loss seed sets: {'baseline': [42, 2024, 2025, 2026, 3407], 'weighted_bce': [42, 2024, 2025, 2026, 3407], 'focal': [42, 2024, 2025, 2026, 3407], 'cb_focal': [42, 2024, 2025, 2026, 3407]}.
- Deterministic errors: none.

WARN is retained even when deterministic checks pass because the historical scalers include internal-validation data and the required fresh-agent semantic audit could not be performed under the active no-delegation constraint.
