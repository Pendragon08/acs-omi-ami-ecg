# ECG-LFM Local-Best LoRA/DoRA 优化实验报告

> 最后更新：2026-08-28T19:49:11+00:00。本文只汇总已落盘且可追溯的实验结果；未完成方法不会被当作结论。

## 1. 结论摘要

当前已完成方法：Baseline（冻结卷积前端的部分参数微调）、LoRA、DoRA。按五折验证集全局 OOF 的 macro-AUPRC 优先、macro-AUROC 与 macro-F1 次级排序，当前最佳为 **LoRA**。

本实验的比较因素只有适配策略：Baseline 为冻结卷积特征提取器、更新 Transformer 上下文编码器和分类头的部分参数微调；LoRA 与 DoRA 从同一官方 ECG-LFM 预训练权重开始；不引入采样器、Focal/CB-Focal、数据增强或网络结构改动。

## 2. 数据与不可变设置

- 任务：四个相互独立的 sigmoid 二分类输出 STEMI/NSTEMI/UA/OMI；保留明确的 `[0,0,0]` non-ACS 样本，仅剔除标签未知样本。
- 内部队列：17,960 条 ECG、17,018 名患者；外部盲测队列：1,995 条 ECG。每条记录使用两个 5 秒片段，先平均 logits 再 sigmoid。
- 划分：沿用历史 patient-level 五折划分（seed=2026），不重新随机划分；每折训练集约 14,300 条记录、验证集约 3,600 条记录。
- 预处理：官方 ECG-LFM 500 Hz 原始 ECG 管线；normalize/filter/padding 均保持 baseline 配置。验证/外部集不 oversampling。
- 训练：官方 Conformer ECG-LFM（24 层上下文编码器，1024 hidden，16-head relative-position attention）+ 4 输出线性分类头；Adam（betas 0.9/0.98），effective batch 256（physical batch 32、update_freq 8），最多 100 epoch，validation loss checkpoint，patience=10，step decay=0.99。
- 损失：四组对比中本报告仅使用项目原始 `BCEWithLogitsLoss`（unweighted）；不使用 WeightedRandomSampler，不重复叠加 `pos_weight`。
- 阈值：仅在内部 OOF 验证预测上逐标签网格搜索 0.01–0.99 的 Youden J；固定后应用到外部集，外部标签不参与任何选择。

## 3. 适配策略

| 方法 | 目标模块 | rank | alpha | dropout | 学习率 | 可训练参数 |
|---|---|---:|---:|---:|---:|---:|
| Baseline (partial-FT) | 冻结卷积特征提取器；更新上下文编码器、投影/位置层、LayerNorm 和分类头 | — | — | — | 1e-6 | 见下表 |
| LoRA | 24 层 × (q/k/v/out/pos + FFN1/FFN2 两层) | 8 | 16 | 0.05 | 1e-4 | 适配器 + 分类头 |
| DoRA | 同 LoRA 目标；额外逐输出行 magnitude | 8 | 16 | 0.05 | 5e-5 | 适配器 + magnitude + 分类头 |

适配器注入代码位于 `third_party/ECG-LFM/fairseq_signals/modules/parameter_efficient.py`，模型集成位于 `third_party/ECG-LFM/fairseq_signals/models/classification/ecg_transformer_classifier.py`。默认 `adapter_method=none`，因此历史 baseline 行为保持不变。

## 3.1 参数高效微调与训练成本分析（方法创新点）

需要明确：本报告中的 Baseline 不是字面意义上的“全模型全参数微调”。配置 `feature_grad_mult=0.0` 时，官方 ECG-LFM 代码在 `get_embeddings()` 中使用 `torch.no_grad()` 计算卷积特征提取器；因此卷积前端不接收梯度。`freeze_finetune_updates=0` 则表示从第一个更新开始，Transformer 上下文编码器、post-extract projection、位置卷积、LayerNorm 和四标签分类头参与微调。项目中真正的 head-only 方案是 `linear_probe.yaml`，本报告没有把它当作 Baseline。

| 方法 | 冻结部分 | 更新部分 | 有效可训练参数 | 总参数 | 可训练比例 | 相对 Baseline 的有效更新参数 |
|---|---|---|---:|---:|---:|---:|
| Baseline (partial-FT) | 卷积特征提取器 | Transformer 上下文编码器、投影/位置层、LayerNorm、分类头 | 622,479,108 | 622,880,004 | 99.94% | 1.0× |
| LoRA | 卷积前端 + 全部预训练上下文参数 | LoRA 低秩适配器 + 分类头 | 5,902,340 | 628,778,244 | 0.94% | **约 105.5× 更少** |
| DoRA | 卷积前端 + 全部预训练上下文参数 | DoRA 低秩适配器、magnitude + 分类头 | 6,270,980 | 629,146,884 | 1.00% | **约 99.3× 更少** |

Baseline checkpoint provenance 同时记录了 `requires_grad_parameters=622,880,004` 和 `effective_trainable_parameters_at_checkpoint=622,479,108`。前者是因为上游实现没有把卷积前端的 `requires_grad` 标志改为 False；其运行时 `no_grad` 路径仍然阻止前端梯度。后者扣除了 400,896 个前端参数，是本报告用于公平比较的有效训练参数量。

这构成了本方案的参数效率创新点：LoRA/DoRA 将需要梯度、优化器状态和更新的参数从约 6.22 亿降至约 590–627 万，显著降低优化器状态、梯度、checkpoint 和训练通信开销。LoRA 相对 Baseline 同时取得 macro-AUPRC `+0.0076`、macro-AUROC `+0.0036` 和 macro-F1 `+0.0085`，Specificity 仅变化 `-0.0016`，形成较好的性能—训练成本折中。

参数量下降不应直接等同于总显存或墙钟时间按相同比例下降：冻结主干权重仍需加载，长 ECG 序列的中间激活仍需计算。本轮没有注册相同 batch/GPU/数据量下的 peak-VRAM、能耗和墙钟时间配对测量，因此报告只宣称“有效更新参数和优化器状态规模显著下降”，不夸大为总显存下降 99%。论文定稿前应补做固定 GPU 的资源基准。

## 4. 五折验证集指标（global OOF Youden 阈值；mean ± std）

### Baseline (partial-FT)

| Label | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STEMI | 0.9463 ± 0.0079 | 0.6852 ± 0.0325 | 0.8814 ± 0.0160 | 0.8863 ± 0.0143 | 0.4053 ± 0.0297 | 0.9885 ± 0.0015 | 0.5548 ± 0.0285 | 0.5496 ± 0.0276 | 0.8859 ± 0.0132 |
| NSTEMI | 0.8008 ± 0.0250 | 0.2648 ± 0.0277 | 0.7505 ± 0.0200 | 0.7179 ± 0.0229 | 0.1649 ± 0.0136 | 0.9749 ± 0.0024 | 0.2702 ± 0.0194 | 0.2559 ± 0.0250 | 0.7201 ± 0.0221 |
| UA | 0.6711 ± 0.0036 | 0.4830 ± 0.0070 | 0.7210 ± 0.0415 | 0.5235 ± 0.0456 | 0.4451 ± 0.0094 | 0.7809 ± 0.0119 | 0.5498 ± 0.0073 | 0.2350 ± 0.0114 | 0.5918 ± 0.0162 |
| OMI | 0.9051 ± 0.0091 | 0.4110 ± 0.0268 | 0.8445 ± 0.0257 | 0.8246 ± 0.0151 | 0.2486 ± 0.0122 | 0.9873 ± 0.0019 | 0.3839 ± 0.0134 | 0.3972 ± 0.0128 | 0.8259 ± 0.0131 |
| macro | 0.8308 ± 0.0078 | 0.4610 ± 0.0108 | 0.7993 ± 0.0143 | 0.7380 ± 0.0134 | 0.3160 ± 0.0113 | 0.9329 ± 0.0029 | 0.4397 ± 0.0117 | 0.3594 ± 0.0111 | 0.7559 ± 0.0093 |

OOF Youden thresholds: STEMI=0.06, NSTEMI=0.07, UA=0.35, OMI=0.05.

## 5. 汇总对比（macro 五折 mean ± std）

| Method | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy | Trainable / total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (partial-FT) | 0.8308 ± 0.0078 | 0.4610 ± 0.0108 | 0.7993 ± 0.0143 | 0.7380 ± 0.0134 | 0.3160 ± 0.0113 | 0.9329 ± 0.0029 | 0.4397 ± 0.0117 | 0.3594 ± 0.0111 | 0.7559 ± 0.0093 | 622,479,108 / 622,880,004 (99.94%) |
| LoRA | 0.8344 ± 0.0069 | 0.4686 ± 0.0081 | 0.8030 ± 0.0370 | 0.7364 ± 0.0343 | 0.3256 ± 0.0244 | 0.9347 ± 0.0070 | 0.4481 ± 0.0255 | 0.3677 ± 0.0221 | 0.7564 ± 0.0276 | 5,902,340 / 628,778,244 (0.94%) |
| DoRA | 0.8339 ± 0.0046 | 0.4634 ± 0.0080 | 0.8025 ± 0.0178 | 0.7420 ± 0.0138 | 0.3220 ± 0.0133 | 0.9350 ± 0.0020 | 0.4488 ± 0.0130 | 0.3680 ± 0.0114 | 0.7632 ± 0.0124 | 6,270,980 / 629,146,884 (1.00%) |

相对 Baseline 的当前最佳方法（LoRA）变化：

| Metric | Baseline mean | Best method mean | Δ |
|---|---:|---:|---:|
| AUROC | 0.8308 | 0.8344 | +0.0036 |
| AUPRC | 0.4610 | 0.4686 | +0.0076 |
| SENSITIVITY | 0.7993 | 0.8030 | +0.0037 |
| SPECIFICITY | 0.7380 | 0.7364 | -0.0016 |
| PPV | 0.3160 | 0.3256 | +0.0096 |
| F1 | 0.4397 | 0.4481 | +0.0085 |
| MCC | 0.3594 | 0.3677 | +0.0082 |
| ACCURACY | 0.7559 | 0.7564 | +0.0004 |

## 6. 逐折门禁与训练曲线

下表来自各折 `curves/overfitting_audit.json`；训练/验证 loss、AUROC、AUPRC、F1 曲线 PNG/CSV 均保存在对应 `fold*/curves/`。

### Baseline (partial-FT)

| Fold | Best validation-loss epoch | Best validation loss | Last epoch | Possible overfit |
|---:|---:|---:|---:|---|
| 0 | 5 | 0.412 | 15 | True |
| 1 | 5 | 0.41 | 15 | True |
| 2 | 5 | 0.401 | 15 | True |
| 3 | 6 | 0.406 | 16 | True |
| 4 | 5 | 0.402 | 15 | True |

## 7. 外部盲测提交

外部集使用五折模型概率算术平均，并固定内部 OOF 阈值生成四标签 hard-label CSV；提交文件、原始 HTML 回包与 JSON 指标均放在最佳方法的 `summary/`。平台返回的是 hard-label 指标，不提供可复算的外部 AUROC/AUPRC；外部标签未在本地读取。

LoRA 已完成外部平台提交并返回四标签 hard-label 指标；DoRA 已生成外部预测与提交 CSV，但尚未提交平台。

| Label | Sensitivity | Specificity | PPV | NPV | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|
| STEMI | 0.8827 | 0.9051 | 0.4511 | 0.9887 | 0.9033 | 0.5971 |
| NSTEMI | 0.7794 | 0.7009 | 0.1601 | 0.9775 | 0.7063 | 0.2657 |
| UA | 0.7478 | 0.5042 | 0.4393 | 0.7938 | 0.5875 | 0.5534 |
| OMI | 0.8984 | 0.8163 | 0.2511 | 0.9915 | 0.8216 | 0.3925 |

## 8. 结论与可追溯文件

- 当前最佳方法：**LoRA**（按已完成方法的 macro-AUPRC 优先选择）。LoRA/DoRA 五折已完成；DoRA 尚无外部平台回包，因此外部泛化不能做完整适配器间比较。
- 强/中等/无提升的判定应以 Baseline 对比、五折均值与标准差共同决定，不能依据单折或 Accuracy 单项。
- 训练命令、seed、折号、官方仓库 commit、配置哈希与工作树差异保存在每折 `launch_provenance.json`。
- Baseline OOF：`results/ecg_lfm_4label/finetune_paper/summary/oof_evaluation.json`。
- LoRA/DoRA OOF：`results/ecg_lfm_local_best/{lora,dora}/summary/oof_evaluation.json`。
- 五折预测、阈值、逐折指标、混淆矩阵、训练曲线和外部提交文件均与上述目录同层保存。
- 报告生成脚本：`C:/Users/Pende/Desktop/基于现有 ACS 预处理数据的 TimesNet OMI-AMI 多标签分类实施方案/scripts/write_ecg_lfm_local_best_report.py`。

### 配置哈希

- Baseline：`C:/Users/Pende/Desktop/基于现有 ACS 预处理数据的 TimesNet OMI-AMI 多标签分类实施方案/configs/ecg_lfm/finetune_paper.yaml` (SHA-256 `3c251ace66f33076b442b4265c13c0acad3dcf56a1c0fc10752f5c341aa5953f`)
- Local-best baseline copy：`C:/Users/Pende/Desktop/基于现有 ACS 预处理数据的 TimesNet OMI-AMI 多标签分类实施方案/configs/ecg_lfm/local_best_baseline.yaml` (SHA-256 `fec3d966c169de09794053ded0de6a09d5cbcd879c3fd2b7c3a934d1fcd277ae`)
- LoRA：`C:/Users/Pende/Desktop/基于现有 ACS 预处理数据的 TimesNet OMI-AMI 多标签分类实施方案/configs/ecg_lfm/lora.yaml` (SHA-256 `131ddd1afa5224b0e2c5bd3ebf20c3cc98f4bf560a48adf79124840b88a09540`)
- DoRA：`C:/Users/Pende/Desktop/基于现有 ACS 预处理数据的 TimesNet OMI-AMI 多标签分类实施方案/configs/ecg_lfm/dora.yaml` (SHA-256 `527aeea7a78b444c2d09e299f67b5a2a6c2d55a98ac7fef686a4755d8f279387`)

