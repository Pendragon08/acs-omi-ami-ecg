# ECG-LFM 当前 Joint-6 与历史联合任务实现

_更新于 2026-09-07 · 当前入口是 Joint-6；旧双 Head 仅作历史与消融参照_

---

## 🎯 当前 Joint-6

`ECG-LFM → LoRA → 1024-D pooled feature → 6 logits → Softmax`，训练使用单一 CrossEntropyLoss。

联合类别依次为 `OTHER / STEMI+nOMI / STEMI+OMI / NSTEMI+nOMI / NSTEMI+OMI / UA`。ACS 概率通过 `[q0, q1+q2, q3+q4, q5]` 恢复并 argmax；OMI 概率为 `q2+q4`，阈值在 inner-validation 选择后冻结到 outer-test。不是六个独立二分类，也不是两个 Head。

## ⚙️ 当前实现入口

| 文件 | 用途 |
|---|---|
| [joint6.py](joint6.py) | 六类标签映射与概率边缘化 |
| [audit_joint6_labels.py](audit_joint6_labels.py) | 原始标签、例外样本和患者划分审计 |
| [run_joint6_fold.py](run_joint6_fold.py) | 单个 nested fold 的 ECG-LFM 训练 |
| [evaluate_joint6_fold.py](evaluate_joint6_fold.py) | Joint-6、ACS、OMI 与 NSTE 子组评价 |
| [summarize_joint6_cv.py](summarize_joint6_cv.py) | 七种模型的逐折、均值/标准差与 pooled 汇总 |
| [training_reporting.py](training_reporting.py) | 指定折、最终完整运行的 epoch-end 曲线解析 |
| [repair_joint6_reporting.py](repair_joint6_reporting.py) | 保留旧指标和 receipt 的报告修复 |
| [joint6_optimization.py](joint6_optimization.py) | 当前 Fold0 inner-only 学习率筛选，不访问 outer-test |

当前配置：[LoRA](../../configs/ecg_lfm/corrected_joint6_lora.yaml)、[Partial FT](../../configs/ecg_lfm/corrected_joint6_partial.yaml)。LoRA 为 r=8、alpha=16、dropout=0.05；完整设置以 YAML 和运行凭据为准。旧五折主结果见[结果目录](../../results/joint6_cv/README.md)，在研优化见[优化报告](../../docs/ECG_LFM_Joint6_Optimization_Report.md)。

## 📚 保留的历史双 Head

旧 `factorized_two_head` 使用 ACS 四状态 Softmax + OMI 独立 Sigmoid，损失为 CE+BCE。其 `run_joint_fold.py`、`evaluate_joint.py` 与 `corrected_joint_{baseline,lora,dora}.yaml` 保留兼容性；其中共享工具仍可能被当前代码引用，不能按旧文件名直接删除。

历史结果见[修正评估报告](../../docs/ECG_LFM_corrected_evaluation_report.md)。该历史 pooled OOF 协议与当前 nested Joint-6 不完全相同，不能直接作为任务形式优劣的因果证据。DoRA/RDR/MOO/RL 不属于当前 Joint-6 LoRA 主方案。

## 🔐 复现边界

原始 ECG、患者级表格、checkpoint、逐例预测和提交 CSV 不进入 GitHub。当前 Joint-6 没有独立外部测试结果。此前本文件将双 Head 描述为“当前方案”的表述已更正，论文写作请从[中文文档导航](../../docs/README.md)开始。
