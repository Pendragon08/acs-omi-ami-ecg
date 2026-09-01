# W0–W6 Nested Fold0 类别重加权实验

## 实验边界

- 模型：ECG-LFM LoRA，rank 8 / alpha 16 / dropout 0.05；
- split：相同 patient-level nested Fold0；
- seed：2026；
- 只改变训练阶段 loss weighting；
- checkpoint selection：统一未加权 inner-validation CE+BCE；
- outer-test：模型与 OMI threshold 冻结后评价一次。

## Primary endpoints

| 方法 | ACS Macro AUPRC | NSTEMI AUPRC | OMI AUPRC | 判定 |
|---|---:|---:|---:|---|
| W0 | **0.546685** | **0.287459** | **0.412660** | 最佳 |
| W1 | 0.530770 | 0.279219 | 0.409730 | 未超过 W0 |
| W2 | 0.517696 | 0.247335 | 0.381572 | 未超过 W0 |
| W3 | 0.495348 | 0.240624 | 0.361160 | 未超过 W0 |
| W4 | 0.534947 | 0.263176 | 0.388518 | 未超过 W0 |
| W5 | 0.534947 | 0.263176 | 0.388518 | 未超过 W0 |
| W6 | 0.534947 | 0.263176 | 0.388518 | 未超过 W0 |

最终判定：**RDR NO-GO**。不执行完整五折、多 seed 和 paired bootstrap。

## 文件说明

- `fold0_comparison.csv`：W0–W6 完整聚合指标；
- `fold0_runtime.csv`、`fold0_rdr_runtime.csv`：训练时间与峰值显存；
- `rdr_profile_fold0.json`：canonical 1024-D RDR profiling；
- `rdr_warmup_selection.json`：仅由 W0 inner-validation 冻结的 warmup；
- `source_unit_tests.txt`：门控时的 11 项单元测试结果；当前扩展测试套件共 15 项并已全部通过；
- `w*/fold0/launch_provenance.json`：每组配置、权重、source/split/checkpoint hash；
- `../adaptive_weighting_evaluation/w*/fold0/`：聚合 metrics、图表与训练曲线。
- `../adaptive_weighting_evaluation/w0/fold0/calibration.json`：W0 的 inner-fit / outer-evaluate 校准结果；
- `../single_task_vs_joint_evaluation/`：ACS-only / OMI-only / Joint W0 对照；
- `artifact_manifest.json`：可公开聚合 artifact 的 SHA-256，并明确列出受保护或仅 dry-run 文件。

患者级 predictions、原始日志流和 checkpoint 未纳入 Git。
