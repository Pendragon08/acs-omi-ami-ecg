# ECG-LFM 校正联合任务实验报告

更新时间：2026-08-30 10:52（UTC）
状态：ECG-LFM Baseline 五折已完成并生成 pooled OOF 验证汇总；LoRA
fold0--4 已完成并生成 pooled OOF 汇总；DoRA fold0--2 已完成。Fold3 于
2026-08-30 09:34 UTC 训练到 epoch 10（valid_loss=1.03139）并写出
`checkpoint_best.pt`，随后进程收到 SIGKILL，未正常收尾；fold4 尚未启动，
DoRA pooled OOF 汇总尚未生成。LoRA/DoRA 最终比较和外部平台提交需等待 DoRA
恢复并完成五折。

## 任务定义

ECG-LFM 当前使用五个输出：ACS 四分类
`OTHER / STEMI / NSTEMI / UA`，以及独立 OMI/NOMI 二分类。STEMI、NSTEMI、
UA 在 ACS 头中使用交叉熵和 softmax-argmax，不能同时为阳性；OMI 使用稳定的
`BCEWithLogitsLoss` 和验证集确定的阈值。`[0,0,0]` 保留为 `OTHER`，而不是
缺失值。该定义与 E4、传统架构完全一致，保证比较只反映编码器/微调方式差异。

## 代码与错误修复记录

首次校正运行在 Fairseq 验证阶段暴露真实的形状错误：源标签是四列
`[STEMI,NSTEMI,UA,OMI]`，而模型 `num_labels=5`，验证循环 reshape 失败。
该运行没有有效指标，不能作为结果。现在的
`experiments/ecg_lfm/run_joint_fold.py` 会在受控执行环境生成并审计私有的
五列 `[OTHER,STEMI,NSTEMI,UA,OMI]` 标签文件；criterion
`third_party/ECG-LFM/fairseq_signals/criterions/acs_joint.py` 同时兼容源四列
检查格式和 Fairseq 五列格式，并对非二值标签、亚型重叠、缺失 OTHER 状态直接
报错。修复后 fold0 的验证 reshape smoke test 已通过。

## 固定实验配置

| 项目 | 配置 |
|---|---|
| 划分 | 历史患者级五折，seed=2026，不重新随机划分 |
| 适配方式 | Baseline、LoRA、DoRA，各 5 折 |
| GPU | Baseline/LoRA 使用物理 GPU1；DoRA 使用物理 GPU0（均单卡绑定） |
| batch | 64；`update_freq=4`，有效 batch size=256 |
| 优化器 | Adam，沿用各适配方式登记的学习率 |
| 损失 | ACS CE + OMI 稳定 BCE，未叠加额外 `pos_weight` |
| 训练上限 | 100 epoch，Fairseq patience=10，逐 epoch 验证 |
| 评估 | ACS 四分类、ACS 阳性条件三分类、OMI 二分类；OMI 阈值只在验证 OOF 上确定 |

运行根目录为 `results/corrected_ecg_lfm_joint_bs64_v2/`，原始日志目录为
服务器上的 `logs/corrected_ecg_lfm_joint_bs64_v2/`。训练脚本使用 screen，并且
只绑定一个 GPU。

## 当前进度（报告更新时间）

Baseline fold0--4 已完成，统一评估器已经生成五折 pooled OOF 验证结果、阈值
分析、ROC/PR 曲线和混淆矩阵。各折最佳验证损失如下：fold0 `1.06511`
（epoch 7）、fold1 `1.05344`（epoch 4）、fold2 `1.04459`（epoch 5）、
fold3 `1.04264`（epoch 5）、fold4 `1.04088`（epoch 6）。五折最佳损失均值为
`1.04933±0.01006`；ACS 验证准确率为 `0.56563±0.00508`，OMI 验证准确率为
`0.93697±0.00150`。LoRA fold0--4 已完成，已生成 `lora/summary/summary.json`
（pooled OOF）。DoRA fold0--2 已正常完成；fold3 训练至 epoch 10、valid_loss
`1.03139` 后在保存 checkpoint 后被 SIGKILL，现有 `checkpoint_best.pt` 约 2.4 GB，
但该折没有正常完成标记；fold4 尚未启动。DoRA 五折完成后再生成最终比较、曲线
和外部平台提交结果。

## 三种微调方式训练成本对比

参数量取各方式 fold0 的 Fairseq 启动日志；训练时间为各折日志中的
`done training in`（仅训练，不含统一评估），GPU 内存取训练日志记录的最小
`gb_free`。Baseline/LoRA 的时间为五折均值±标准差，DoRA 目前只有 fold0，
因此不外推五折均值。

| 方法 | 模型总参数 | 可训练参数 | 可训练比例 | 记录的最小剩余显存 | 估计峰值显存 | 单折训练时间 | 五折训练时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline（全参） | 622.881 M | 622.881 M | 100.000% | 12.0 GB | ≈35.4 GB | 6670.3±495.7 s | 33351.3 s（9.26 h） |
| LoRA（rank=8） | 628.779 M | 5.903 M | 0.9389% | 19.0 GB | ≈28.4 GB | 6131.3±367.1 s | 30656.7 s（8.52 h） |
| DoRA（rank=8） | 629.148 M | 6.272 M | 0.9969% | 16.1 GB | ≈31.3 GB | 7404±1615 s（已完成 fold0--2） | 22213.1 s（已完成 3 折，fold3--4 待运行） |

因此，当前日志直接支持的结论是：LoRA 将可训练参数降至不到 1%，峰值显存
较 Baseline 约少 7.0 GB；DoRA 已完成 3 折，较 LoRA 多约 2.9 GB 峰值显存，
当前单折耗时均值也更长，最终五折成本需等 fold3--4 结束后再比较。上述显存为日志估计
值，不替代 `nvidia-smi` 的瞬时峰值测量。

## Baseline 五折验证指标（pooled OOF）

以下结果来自五个历史患者级验证折的 OOF 预测（`n=17,960`），不是外部测试集。
ACS 四分类（`OTHER/STEMI/NSTEMI/UA`）不受 OMI 阈值影响：

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.56843 |
| Macro-F1 | 0.49944 |
| Macro-AUROC (OvR) | 0.78255 |
| Macro-AUPRC (OvR) | 0.53293 |

ACS 四分类逐类验证结果：

| 类别 | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OTHER | 0.69757 | 0.68730 | 0.72822 | 0.52463 | 0.60982 | 0.65423 | 0.66379 | 0.25840 |
| STEMI | 0.94673 | 0.69112 | 0.67060 | 0.96404 | 0.61947 | 0.97103 | 0.64402 | 0.61218 |
| NSTEMI | 0.81407 | 0.27171 | 0.19757 | 0.97638 | 0.38185 | 0.94279 | 0.26041 | 0.23764 |
| UA | 0.67183 | 0.48159 | 0.38516 | 0.78412 | 0.48549 | 0.70685 | 0.42955 | 0.18044 |

在 ACS 阳性子集（`n=8,890`）上，三亚型分类 Accuracy=`0.80551`、Macro-F1=
`0.64387`。

OMI 验证指标随阈值变化如下；AUROC/AUPRC 与阈值无关：

| 阈值策略 | threshold | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 固定 0.5 | 0.50 | 0.90523 | 0.41586 | 0.23545 | 0.98596 | 0.53452 | 0.94958 | 0.32690 | 0.32739 | 271/16573/236/880 |
| 最大 F1 | 0.18 | 0.90523 | 0.41586 | 0.61599 | 0.92992 | 0.37573 | 0.97250 | 0.46675 | 0.43600 | 709/15631/1178/442 |
| Youden J（当前选择） | 0.05 | 0.90523 | 0.41586 | 0.84883 | 0.82277 | 0.24697 | 0.98758 | 0.38261 | 0.39689 | 977/13830/2979/174 |
| Sensitivity≥0.90 | 0.03 | 0.90523 | 0.41586 | 0.90096 | 0.75692 | 0.20242 | 0.99112 | 0.33057 | 0.35683 | 1037/12723/4086/114 |
| Sensitivity≥0.95 | 0.01 | 0.90523 | 0.41586 | 0.96004 | 0.58653 | 0.13718 | 0.99536 | 0.24006 | 0.26915 | 1105/9859/6950/46 |

阈值只使用验证 OOF 确定，未使用外部测试标签。当前 `summary.json`、阈值分析、
训练/验证曲线、ROC/PR 曲线和混淆矩阵均位于
`results/corrected_ecg_lfm_joint_bs64_v2/baseline/summary/`。

## LoRA 五折验证指标（pooled OOF）

LoRA 五个历史患者级验证折均已完成；以下结果来自服务器上的
`lora/summary/summary.json`，`n=17,960`，不是外部测试集。ACS 使用四分类
argmax，OMI 的 AUROC/AUPRC 与阈值无关。

| 任务 | Accuracy | AUROC | AUPRC | Macro-F1 |
|---|---:|---:|---:|---:|
| ACS 四分类（OTHER/STEMI/NSTEMI/UA） | 0.56949 | 0.78705 | 0.53897 | 0.51026 |
| ACS 阳性条件三分类（`n=8,890`） | 0.80585 | — | — | 0.65409 |
| OMI 二分类 | 见下表 | 0.91063 | 0.43153 | 见下表 |

ACS 四分类逐类结果：

| 类别 | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OTHER | 0.70111 | 0.69078 | 0.69978 | 0.56175 | 0.61964 | 0.64714 | 0.65728 | 0.26415 |
| STEMI | 0.94910 | 0.70167 | 0.66158 | 0.96670 | 0.63431 | 0.97035 | 0.64766 | 0.61636 |
| NSTEMI | 0.82297 | 0.28046 | 0.23563 | 0.96753 | 0.34892 | 0.94488 | 0.28130 | 0.24431 |
| UA | 0.67500 | 0.48297 | 0.42427 | 0.76649 | 0.49005 | 0.71568 | 0.45480 | 0.19811 |

OMI 二分类阈值结果：

| 阈值策略 | threshold | Sensitivity | Specificity | PPV | NPV | F1 | MCC | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 固定 0.5 | 0.50 | 0.28236 | 0.98304 | 0.53279 | 0.95239 | 0.36911 | 0.35885 | 325/16524/285/826 |
| Youden J | 0.05 | 0.85491 | 0.83586 | 0.26289 | 0.98825 | 0.40213 | 0.41651 | 984/14050/2759/167 |
| 最大 F1 | 0.25 | 0.55430 | 0.94598 | 0.41268 | 0.96875 | 0.47312 | 0.43683 | 638/15901/908/513 |
| Sensitivity≥0.90 | 0.02 | 0.91920 | 0.74204 | 0.19614 | 0.99260 | 0.32330 | 0.35328 | 1058/12473/4336/93 |

其中，`threshold=0.25` 给出最高 OMI F1；`threshold=0.05` 是当前偏向灵敏度
与 Youden J 的工作点。LoRA 相比 Baseline 的 pooled OOF 增量为：ACS
Macro-AUROC `+0.00450`、Macro-AUPRC `+0.00604`、Macro-F1 `+0.01082`；
OMI AUROC `+0.00540`、AUPRC `+0.01567`。这些增量仍需 DoRA 完成及外部平台
测试后再作最终结论。

## 结果归档计划

每个适配方式完成后将写入：

- `summary/summary.json`、五折 OOF/阈值/外部平台聚合指标；
- 每折训练历史和脱敏的 `training_log_sanitized.csv`（仅 fold、epoch、loss、
  验证指标、学习率）；
- 训练/验证损失与学习率、ROC/PR、混淆矩阵 PNG；
- 外部平台返回的聚合 JSON 与提交 provenance（不含提交 CSV）。

原始 ECG、患者级表格、checkpoint、逐例预测、Fairseq 原始 JSONL 和服务器
路径不会上传 GitHub。Baseline 的可公开脱敏日志和图表已归档；LoRA/DoRA 队列
完成后再追加最终比较表和外部平台逐标签指标，并在
`docs/logs/EXPERIMENT_LOG_INDEX.md` 登记。
