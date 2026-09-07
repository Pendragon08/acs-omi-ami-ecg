# Joint-6 分阶段优化实验报告

## 1. 任务与执行边界

执行批次：`joint6_optimization_20260907`。用户批准按照审查建议顺序实施。

保留 ECG-LFM + LoRA（r=8、alpha=16、dropout=0.05）、六分类 Joint-6、原生 ECG 预处理和现有患者互斥 nested folds。OMI 概率为 q2+q4，ACS 通过概率边缘化恢复后 argmax。

不运行 RL，不重新下载/预处理 ECG，不修改已有划分，不用 outer-test 或 external-test 选择任何参数。历史 outer 结果已经用于发现研究问题，不能把后续在同一队列上的验证宣称为全新、完全未接触的确认性研究；最终独立验证需要冻结方案后进行。

## 2. 阶段顺序与预先确定的规则

1. 修复单折曲线与成本统计。旧曲线备份，原始预测、metrics.json 和 evaluation receipt 不修改。
2. 仅在 nested Fold0 比较 unweighted CE 下的学习率：1e-4（已有基线）、5e-5、3e-5。后两组从相同 pretrained 初始化，不从已微调模型继续训练。batch=64、accumulation=4、Adam、原 StepLR、patience=10、seed=2026 均保持。
3. 学习率结果完整后，先比较选定 LR 下的未加权 CE 与温和静态类别加权 CE。权重仅来自 inner-train；不同时改变增强、采样或模型。具体归一化约定在该阶段启动前记录并测试。
4. 模型候选确定后，比较旧 0.01 阈值网格与 inner 实际分数阈值搜索；对六分类 logits 进行 inner-fit temperature scaling，再边缘化。保留旧协议作为对照，不覆盖旧指标。
5. 仅对筛选通过的候选补完整五折、多 seed（2026/2027/2028）与 patient-level paired bootstrap。不得将五折当成五个 seed。

严格 nested 的补充约束：Fold0 inner-validation 与其他 outer fold 存在样本交叉，不能把 Fold0 筛选的超参数直接复制到其余折后宣称得到无选择偏差的 nested 结果。扩展时需在每个 outer-train 内按预先冻结的同一候选集合/选择规则进行 inner 选择，或明确将固定 Fold0 参数的五折标记为探索性复验。不能把后者冒充独立确认性验证。

学习率筛选只读取各候选的 inner-validation。每组 checkpoint 均由 minimum inner-validation Joint CE 选出。主排序指标为 Joint-6 Macro AUPRC；同时完整展示 ACS/OMI/NSTE 指标，不挑单项最好的一次。晋级候选相对当前基线的 OMI AUPRC 和 NSTE-ACS OMI AUPRC 均不得下降超过 0.005，Youden OMI sensitivity 不得下降超过 0.02。满足上述预先设定的工程筛选容差后，按 Joint-6 Macro AUPRC 最大选择；相同分数优先保留基线。没有通过者则保留 1e-4，仍可按顺序开展静态加权对比。这些容差不是临床非劣效界值，也不构成显著性检验。

## 3. 报告修复说明

发现历史单折曲线混入五折，JSON logger 重复输出相同 epoch，且遗漏 train_lr。修复后每折只保留本折 epoch-end train/validation 摘要，同一次运行内的冲突记录报错而不是静默合并。LoRA Fold4 原始日志含一次未完成尝试及一次完整重跑，按明确的配置起始标记和完成标记分隔，只使用最终完整运行；旧日志全部保留，并记录 discarded_prior_log_runs=1。保存 checkpoint 的 iterator epoch 指向下一轮，校验改为比较真实 num_updates 和 val_loss，避免把游标误当最佳 epoch。

Partial FT 的 requires_grad 参数统计忽略了 feature_grad_mult=0 的 no_grad 卷积前端。新统计分开报告 requires_grad 与 forward 中有效更新参数；旧 metrics 不重写，使用带源文件 hash 的 reporting_corrections.json 补充。

历史 ECG-LFM 显存为整卡 nvidia-smi 采样，传统模型为 PyTorch allocated 峰值；耗时范围也不同。历史成本不能直接视作统一口径的比较。新实验将明确记录测量范围，不反推历史缺失数值。

原汇总器只认可当前 evaluator 文件 hash，因此纯报告修复也会使历史结果全部被拒绝。已补充精确源版本兼容声明：只允许经审查的旧/新 evaluator hash 组合，并固定依赖文件 hash；未知版本仍拒绝。概率与 metric 函数经 AST 比较未改变，所有预测、指标、checkpoint、assignment 与原 receipt 的校验仍保留，未伪造或改写历史 receipt。完整性校验失败时，汇总器现在先报错，不再覆盖已有汇总。

## 4. 执行与测试记录（启动快照）

本节记录启动时事实；候选的最新完成状态以第 5 节自动汇总及各目录 run_state.json 为准。

报告修复已完成：10 个 ECG-LFM 折已修复曲线与参数口径；全部 35/35 历史实验通过原始凭据和报告版本兼容检查，预测和 metrics.json 未改变。

2026-09-07 11:25（北京时间）确认采用 v2 资源监测的两个候选均开始第 1 轮训练：GPU0 为 lr5e5，GPU1 为 lr3e5，均为单卡约 30 GiB 占用。学习率候选尚未完成；静态加权、校准、追加五折和多 seed 尚未启动，不会仅因任务退出就自动跳过阶段判断。这里没有新性能提升结论。

测试结果：38 项通过（报告/筛选/显存监测 15 项，Joint-6 17 项，历史 corrected evaluator 6 项）；另已完成 seeded CUDA kernel witness、两候选 dry-run 及真实模型训练启动核验。测试明细保存在 `docs/logs/Joint6_Optimization_Verification_20260907.txt`。

资源监测 v1 尝试记录：首次启动在 11:16，随后发现 fairseq Trainer 每步内部会重置 CUDA 峰值计数，外部读取仅得到约 2.8 GiB 的 post-step allocated 值，不能作为峰值。两次尝试因测量代码缺陷中止，不是根据性能指标淘汰；最后一次刷新的记录均为 100 updates，日志及 checkpoint 全部迁移保存在服务器 `results/joint6_optimization_20260907/attempts/telemetry_v1/`，原日志以 `*.telemetry_v1_aborted.log` 保留。没有删除旧实验。

v2 在底层重置计数之前采集峰值，不修改梯度或优化器计算，并从同一 pretrained 与 seed 完整重启两组。已在各 25 updates 验证 allocated 峰值 29045.89 MiB、reserved 峰值 30702.00 MiB；原来的低估值不得进入成本对比。此时每次更新约 3.3 秒，当前阶段预计约 1.5–3 小时，最终由实际收敛和原 early stopping 决定，不是全部后续实验的预计完成时间。

| 实验 | screen 会话 | 日志 |
|---|---|---|
| LoRA LR=5e-5 | joint6_lr5e5_gpu0_20260907 | logs/joint6_optimization_20260907/lr5e5.log |
| LoRA LR=3e-5 | joint6_lr3e5_gpu1_20260907 | logs/joint6_optimization_20260907/lr3e5.log |

两组从同一 pretrained 重新初始化，无恢复 checkpoint；各自训练结束后自动做 inner-only inference、导出完整指标与曲线并刷新本报告。基线复用 hash 校验通过的历史 inner 预测，避免重复训练。其他阶段没有自动训练队列，需在学习率结果齐全并审查后继续。

机器可读结果将保存在 `results/joint6_optimization_20260907/`；受控原始日志位于服务器 `logs/joint6_optimization_20260907/`。checkpoint、record/patient 级预测与原始 ECG 不进入 Git。主报告将随各阶段真实完成状态更新。

## 5. 学习率实验实时汇总

<!-- RESULTS_START -->
| 候选 | LR | 状态 | Joint-6 AUPRC | ACS AUPRC | OMI AUPRC | OMI Sensitivity | NSTE OMI AUPRC |
|---|---:|---|---:|---:|---:|---:|---:|
| baseline_lr1e4 | 0.0001 | complete | 0.394075 | 0.559426 | 0.432049 | 0.813472 | 0.314605 |
| lr5e5 | 5e-05 | training | — | — | — | — | — |
| lr3e5 | 3e-05 | training | — | — | — | — | — |

以上全部是 inner-validation 筛选指标，不是 outer/外部测试，不代表独立泛化性能。

选择状态：`{"status": "pending", "missing": ["lr3e5", "lr5e5"]}`。

<!-- RESULTS_END -->
