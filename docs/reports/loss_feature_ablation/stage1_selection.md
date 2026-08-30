# 第一阶段验证集筛选记录（历史 OMI/AMI 任务）

> 本阶段筛选的是旧 OMI/AMI 二分类实验，当前 ACS/OMI 联合任务不使用这些
> 阈值或模型选择结论。

> 中文说明：本阶段只在固定验证流程中筛选候选 loss/特征。最终结论以五种子
> 均值、标准差和配对统计比较为准，不能把单个 seed 的最高值当作最终结果。

Selection rule: lexicographic_AUPRC_then_F1-Sensitivity-hmean_then_F1_then_Sensitivity_then_Specificity. External test labels/results were not used.

## best_focal

- alpha=0.25, beta=None, gamma=2.0
- AUROC=0.881419, AUPRC=0.344125
- harmonic threshold=0.25, Sens=0.680851, Spec=0.923759, F1=0.445993

## best_cb_focal

- alpha=None, beta=0.99, gamma=1.0
- AUROC=0.882840, AUPRC=0.343715
- harmonic threshold=0.20, Sens=0.776596, Spec=0.869976, F1=0.377261

## Best added feature: OMI

- mode=raw_clin
- AUPRC=0.348832; delta vs controlled raw=+0.003098
- AUROC=0.878458; delta vs controlled raw=-0.000503

## Best added feature: AMI

- mode=raw_age
- AUPRC=0.644271; delta vs controlled raw=+0.002927
- AUROC=0.879029; delta vs controlled raw=+0.001603
