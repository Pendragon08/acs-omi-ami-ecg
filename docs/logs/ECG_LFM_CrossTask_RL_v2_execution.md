# Cross-task Clinical RL v2 执行日志（整理版）

> 该文件是脱敏后的执行索引，不包含患者级预测、原始 ECG、checkpoint 或完整 raw JSONL。完整 raw log 仍保留在受控服务器。

| 阶段 | 时间/状态 | 证据 |
|---|---|---|
| 代码同步 | 2026-09-01，完成 | 实际运行目录 `/data/code/ouyanghao/oyh/ACS-OMI-AMI` |
| 单元测试 | PASS | `tests/test_ecg_lfm_clinical_rl.py`，20/20 |
| smoke | PASS | GPU0/GPU1，各 3 updates；ranking coefficient 0.01/0.05 可运行 |
| v2A LR=1e-6, target=0.005 | FAIL-SAFE | update 7 连续 KL 越界，hard stop + rollback |
| v2A LR=3e-6, target=0.005 | FAIL-SAFE | update 7 连续 KL 越界，hard stop + rollback |
| v2A LR=1e-6, target=0.010 | PASS | 100 updates；clinical utility=1.057250 |
| v2A LR=3e-6, target=0.010 | PASS | 100 updates；clinical utility=1.056510 |
| 候选选择 | PASS | Safety-first 后选择 `R2A_cross_task_head_lr1e6_tkl010_r0p01` |
| 冻结 | PASS | `frozen_before_outer_test`，checkpoint SHA-256 已记录 |
| outer Fold0 | PASS | 只访问一次；`outer_test_evaluations=1` |
| calibration | PASS | ACS temperature、OMI Platt/Beta/Isotonic 均 inner-fit |
| v2B/完整五折/多 seed | 未执行 | 按报告流程保留为后续条件性实验 |

## 运行入口

```text
configs/ecg_lfm/rl/v2a_head_cross_task_targetkl.yaml
scripts/run_ecg_lfm_v2a_cross_task_head.sh
experiments/ecg_lfm/summarize_cross_task_rl_v2.py
experiments/ecg_lfm/freeze_cross_task_v2_candidate.py
experiments/ecg_lfm/evaluate_cross_task_v2_outer_once.py
```

## 结果位置

```text
results/ecg_lfm_cross_task_rl_v2/
results/ecg_lfm_cross_task_rl_v2_evaluation/
```
