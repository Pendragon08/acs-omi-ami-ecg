# OMI 损失函数消融实验总结

## 中文摘要

本实验严格只改变损失函数，固定数据、患者级划分、预处理、模型、优化器、
学习率、batch、训练轮数和阈值选择。结果为 5 个随机种子的均值±样本标准差。
CB-Focal 的最佳组合是 `beta=0.99, gamma=1.0`，Sensitivity 相对 Baseline
提高，但 AUPRC、F1、PPV 和 Specificity 同时下降；配对 bootstrap 对 AUROC/
AUPRC 的差异不显著，因此结论为 **没有令人信服的整体提升**，不能作为当前
核心创新点。完整脱敏日志、曲线和原始汇总在 `results/loss_feature_ablation/`。

Primary operating point: validation threshold maximizing the harmonic mean of F1 and Sensitivity on the 0.01–0.99 grid. Results are mean ± sample SD over seeds [42, 2024, 2025, 2026, 3407].

| Loss | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.8751 ± 0.0038 | 0.3270 ± 0.0155 | 0.6043 ± 0.0644 | 0.9252 ± 0.0289 | 0.3264 ± 0.0741 | 0.9769 ± 0.0031 | 0.4160 ± 0.0437 | 0.3974 ± 0.0352 | 0.9083 ± 0.0241 |
| Weighted BCE | 0.8822 ± 0.0053 | 0.3128 ± 0.0151 | 0.7234 ± 0.0677 | 0.8765 ± 0.0292 | 0.2510 ± 0.0348 | 0.9829 ± 0.0035 | 0.3696 ± 0.0258 | 0.3732 ± 0.0143 | 0.8684 ± 0.0242 |
| Focal | 0.8753 ± 0.0037 | 0.3277 ± 0.0195 | 0.6787 ± 0.0231 | 0.8968 ± 0.0169 | 0.2709 ± 0.0347 | 0.9805 ± 0.0013 | 0.3861 ± 0.0338 | 0.3798 ± 0.0308 | 0.8853 ± 0.0156 |
| CB-Focal | 0.8765 ± 0.0036 | 0.3246 ± 0.0135 | 0.6787 ± 0.0836 | 0.8934 ± 0.0261 | 0.2662 ± 0.0303 | 0.9805 ± 0.0044 | 0.3790 ± 0.0163 | 0.3742 ± 0.0104 | 0.8821 ± 0.0205 |

Best CB-Focal: beta=0.99, gamma=1.0; mean per-seed threshold=0.2180.
Paired seed AUPRC improvements: 3/5.
Verdict: **No convincing improvement**.

## CB-Focal minus Baseline (estimand note)

The table above is a five-seed mean ± sample SD. The following confidence
intervals are a separate paired patient-bootstrap analysis of the stored
pooled validation predictions (2,000 resamples); they are not confidence
intervals for the five-seed means and must not be compared as if they were the
same point estimate. A bootstrap count of zero is reported conservatively as
`p < 1/2001`.

- ΔAUROC=-0.00179; paired patient bootstrap 95% CI [-0.0111, +0.0092], p=0.685
- ΔAUPRC=-0.00689; paired patient bootstrap 95% CI [-0.0262, +0.0138], p=0.5
- ΔSensitivity=+0.13897; paired patient bootstrap 95% CI [+0.0723, +0.2118], p<1/2001
- ΔSpecificity=-0.05259; paired patient bootstrap 95% CI [-0.0635, -0.0422], p<1/2001
- ΔPPV=-0.06449; paired patient bootstrap 95% CI [-0.1011, -0.0295], p<1/2001
- ΔF1=-0.04167; paired patient bootstrap 95% CI [-0.0820, +0.0014], p=0.06

External hidden-label test results were not used to select loss parameters or thresholds. This report is validation evidence; one frozen external evaluation may follow after the method is finalized.
