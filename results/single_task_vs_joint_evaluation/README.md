# ECG-LFM 单任务与联合学习 Fold0 对照

本目录只保存不含患者标识的聚合指标、脱敏训练曲线和图。逐记录预测与 checkpoint 仅保留在受控服务器，不进入 Git。

## 公平性

三组实验共享预训练 ECG-LFM、LoRA（rank 8、alpha 16、dropout 0.05）、nested patient-level Fold0、seed 2026、optimizer、scheduler、batch size、预处理与 outer-test。Joint 直接复用 W0；ACS-only 只优化四分类 CE，OMI-only 只优化二分类 BCE。checkpoint 由 inner-validation loss 选择；OMI 阈值只由 inner-validation 选择并冻结。

## ACS outer-test

| 方法 | Macro AUROC | Macro AUPRC | Macro F1 | Balanced Acc. | Accuracy | NSTEMI AUPRC | NSTEMI F1 | NSTEMI Sens. | NSTEMI-vs-UA AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Joint W0 | **0.783155** | **0.546685** | **0.525076** | **0.516938** | **0.587077** | **0.287459** | **0.306494** | **0.240816** | **0.526039** |
| ACS-only W0 | 0.777611 | 0.539538 | 0.483992 | 0.492823 | 0.581531 | 0.270281 | 0.220896 | 0.151020 | 0.516048 |

## OMI outer-test

| 方法 | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV | NPV | MCC | Accuracy | Youden 阈值 | NSTE-ACS 内 AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Joint W0 | **0.897364** | **0.412660** | **0.367113** | **0.831169** | **0.815407** | **0.235583** | **0.986027** | **0.378533** | **0.816417** | 0.04 | **0.227752** |
| OMI-only W0 | 0.884419 | 0.358639 | 0.340234 | 0.818182 | 0.795259 | 0.214773 | 0.984593 | 0.349713 | 0.796728 | 0.04 | 0.179534 |

Joint − single 的 primary AUPRC 差值：ACS Macro AUPRC `+0.007147`，NSTEMI AUPRC `+0.017178`，OMI AUPRC `+0.054021`。因此本 Fold0 中没有发现 multi-task interference，反而观察到两个任务方向一致的正迁移；OMI 获益更明显。

## 训练行为与成本

| 方法 | Total params | Trainable params | Trainable % | 最佳 epoch | 训练停止 epoch | Wall time | 物理 GPU 峰值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Joint W0 | 628,779,269 | 5,903,365 | 0.9389% | 5 | 14 | 5,236 s | 31,541 MiB |
| ACS-only W0 | 628,778,244 | 5,902,340 | 0.9389% | 5 | 15 | 5,465 s | 38,369 MiB* |
| OMI-only W0 | 628,775,169 | 5,899,265 | 0.9383% | 3 | 13 | 4,601 s | 31,546 MiB |

`*` ACS-only 运行期间，另一用户在同一物理 GPU 启动约 7.2 GiB 的独立进程；监控 CSV 记录的是整张物理卡总占用，因此 38,369 MiB 不能归因于 ACS-only。共享发生前，本任务的观测占用约 31,078 MiB。训练后段均出现 train loss 继续下降而 validation loss 上升，early stopping 正常保护了最佳 checkpoint。

## 结论边界

这是预注册流程中的 nested Fold0 对照，不是完整五折或多 seed 结论，不能写成统计显著。证据支持“当前 Fold0 下 joint learning 优于 single-task”，但若论文需要把 multi-task learning 作为稳定贡献，仍应扩展至完整 outer 5-fold。
