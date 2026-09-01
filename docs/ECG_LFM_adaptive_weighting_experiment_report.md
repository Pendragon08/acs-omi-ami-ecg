# ECG-LFM 自适应重加权实验报告

> 状态：评估修复、W0–W6 nested Fold0 门控、W0 校准和 single-task vs joint Fold0 均已完成并归档；RDR 判定为 NO-GO。本文档只写入已经由代码、配置、预测或日志验证的事实。

## 1. 任务与审计结论

当前 corrected joint task 是一个共享 ECG-LFM encoder 的双头任务：

- ACS：`OTHER / STEMI / NSTEMI / UA` 四类互斥状态，4 logits、softmax、CrossEntropyLoss；
- OMI：独立于 ACS 状态的单个 logit，sigmoid、BCEWithLogitsLoss；
- 联合目标：`L = L_ACS + 1.0 × L_OMI`；
- 当前工程基线：ECG-LFM LoRA（rank 8、alpha 16、dropout 0.05，分类头可训练）；
- 外层划分：既有 patient-level 5-fold，保持不变；
- 本实验新建的内层划分：每个 outer-train 内按 patient-level 划分 inner-train / inner-validation。

本轮明确禁止 RL，未创建或运行 PPO、GRPO、DAPO 或任何 RL trainer。

## 2. 与所提供审计报告的差异

实际仓库扫描确认了以下需要修复或补充的差异：

1. 旧 evaluator 曾存在把 5 个 logits 一起 softmax 的风险；当前已修复为 ACS 四类 softmax 与 OMI 独立 sigmoid。
2. 旧 pooled OOF 结果主要是描述性复核，并非严格 nested validation；本实验因此保留 outer folds，新增 inner-validation。
3. 历史配置的 `acs_weight=null`、`omi_pos_weight=null`、`omi_loss_weight=1.0` 与扫描事实一致。
4. 旧 Baseline、LoRA、DoRA 均有 Fold0–4 checkpoint，但其训练流程不是本轮新建的 nested protocol，不能直接作为 W0–W6 的因果比较结果。
5. 实施 W0 时进一步发现：静态权重若同时改变 validation loss，会使 checkpoint criterion 与 RDR 不一致。已在 W1 开始前修复为“权重只作用于训练；所有 W0–W6 均使用未加权 inner-validation CE+BCE 选 checkpoint”。W0 本身无权重，行为不受该修复影响。

## 3. Phase 1：评估与 provenance 修复

状态：**通过**。

已完成：

- 对既有 Baseline / LoRA / DoRA 的 Fold0–4 checkpoint 只做重新 inference，未重新训练；
- 每条远端 OOF prediction 包含 `record_id`、`patient_id`、fold、ACS 真值与四类概率、OMI 真值与概率；
- 通过 `record_id` 与 master index 显式 join；
- duplicate prediction = 0；missing prediction = 0；missing label = 0；patient overlap = 0；coverage = 100%；
- 新增三项概率语义单元测试；
- 新增输出 artifact hash manifest，远端含 ID 的 patient-level 表不进入 Git。

描述性 pooled OOF 复核如下。注意：这些数字不是本轮 nested W0–W6 结果。

| 旧方法 | ACS Macro AUPRC | NSTEMI AUPRC | OMI AUPRC |
|---|---:|---:|---:|
| Baseline | 0.5329 | 0.2717 | 0.4159 |
| LoRA | 0.5390 | 0.2805 | 0.4315 |
| DoRA | 0.5369 | 0.2825 | 0.4267 |

证据：`results/revalidated/`、`docs/ECG_LFM_corrected_evaluation_report.md`。

## 4. Phase 2：nested patient-level validation

状态：**通过**。

- 5 个 outer fold 均保留；
- inner split 使用 patient group，所有 W0–W6 共享相同 manifest；
- inner-train、inner-validation、outer-test 三者 patient overlap 均为 0；
- outer-test 不参与 early stopping、checkpoint selection、loss 参数、calibration 或 threshold 选择；
- split assignment 与 provenance 均保存 SHA-256。

Fold0：inner-train 11,493 records / 10,892 patients；inner-validation 2,861 / 2,723；outer-test 3,606 / 3,403。

Fold0 inner-train 类别计数：OTHER 5,816，STEMI 910，NSTEMI 807，UA 3,960；NOMI 10,766，OMI 727。

## 5. Phase 3：静态权重 Fold0

状态：**W0–W3 Fold0 全部完成**。

| 方法 | ACS loss | OMI loss | Fold0 outer ACS Macro AUPRC | Fold0 outer NSTEMI AUPRC | Fold0 outer OMI AUPRC |
|---|---|---|---:|---:|---:|
| W0 | Unweighted CE | Unweighted BCE | 0.546685 | 0.287459 | 0.412660 |
| W1 | Static weighted CE | Unweighted BCE | 0.530770 | 0.279219 | 0.409730 |
| W2 | Unweighted CE | Static pos-weight BCE | 0.517696 | 0.247335 | 0.381572 |
| W3 | Static weighted CE | Static pos-weight BCE | 0.495348 | 0.240624 | 0.361160 |

Fold0 inner-train 派生的 ACS 权重为 `[0.248960, 1.591156, 1.794240, 0.365644]`，均值归一化为 1；OMI `pos_weight=14.808803`。未使用 inner-validation 或 outer-test 标签计算权重。

W0 由 inner-validation loss 选择的最佳 checkpoint 位于 epoch 5（validation loss 1.02768，454 updates）；训练在连续 10 次无改善后结束。epoch 14 的 validation loss 已升至 1.74293，而 training loss 继续下降，说明原有配置存在明显过拟合趋势。outer-test 只在 checkpoint 和阈值冻结后评价一次。W0 的 OMI Youden threshold 由 inner-validation 选为 0.04；outer-test OMI AUROC 0.897364、F1 0.367113，ACS Macro F1 0.525076，ACS Brier 0.517459，OMI Brier 0.046773。

W0 alignment：inner-validation 2,861 条、outer-test 3,606 条；duplicate/missing prediction 均为 0，三组 patient overlap 均为 0。

## 6. Phase 4：官方 RDR 移植与 1024-D profiling

状态：**canonical 1024-D profiling 已通过，W4–W6 已完成**。

- 官方仓库：`https://github.com/GoodMorningPeter/RDR`；
- 固定 commit：`bb55ea9f906f00a2d8938a97543fc232b5d26645`；
- License：MIT；
- 上游默认 warmup：160 / 200 epochs；momentum：0.9；temperature exponent：1.0；
- 保留官方 Gram inverse、RuntimeError 时 pseudoinverse、最小值 `1e-9`、inverse-frequency multiplier、batch mean-one normalization 与 absent-class 行为；
- 数值适配仅包括 float32、显式 finite 检查和 non-finite inverse 的 pseudoinverse fallback，不做隐藏的权重上限裁剪；
- canonical RDR 使用线性头前的真实 pooled 1024-D feature，只影响 loss weighting。

由于 ECG-LFM nested baseline 会在远早于 160 epochs 时 early-stop，照搬 `warmup=160` 将导致 RDR 从未启用。仅依据 W0 Fold0 inner-validation 的最佳收敛 update，将官方 `160/200=80%` 比例映射并冻结为 **363 updates**；outer-test 未参与选择。

Canonical profiling 无 NaN/Inf、无 inverse fallback。8 个 batch 的 RDR 计算时间平均 28.45 ms/batch；ACS 最大权重 13.23，OMI 最大权重 17.81。canonical 方案可以运行，因此未设计或冒充 `RDR-Projected`。

## 7. Phase 4 Fold0 对比

| 方法 | ACS loss | OMI loss | ACS Macro AUPRC | NSTEMI AUPRC | OMI AUPRC |
|---|---|---|---:|---:|---:|
| W4 | RDR-CE | Unweighted BCE | 0.534947 | 0.263176 | 0.388518 |
| W5 | Unweighted CE | RDR-BCE | 0.534947 | 0.263176 | 0.388518 |
| W6 | RDR-CE | RDR-BCE | 0.534947 | 0.263176 | 0.388518 |

W4/W5/W6 的 inner-validation 与 outer-test prediction 文件各自 SHA-256 完全一致。训练日志证明 RDR 在 warmup 后确实启用（例如 W6 epoch 14：NSTEMI mean weight 3.26、OMI mean weight 7.09），但 RDR 生效后的模型没有刷新 warmup 阶段的最佳 checkpoint。因此最终评价的三个 adaptive 分支给出了完全相同的预测，且三个 primary AUPRC 均低于 W0。

## 8. 完整五折、统计与多随机种子

状态：**根据预注册停止条件取消**。

- W0–W6 outer 5-fold：未启动；RDR 未通过 Fold0 gate；
- patient-level paired bootstrap 1,000 次：无五折配对结果，因此不执行；
- W0、W3、best adaptive 多 seed：adaptive NO-GO，因此不执行；
- 本结果是 Fold0 screening evidence，不能声称统计显著，也不能泛化为所有不平衡损失均无效。

## 9. Calibration、阈值与 single-task vs joint

状态：**W0 calibration 与 single-task vs joint Fold0 对照均已完成。**

Fold0 gate 确认 W0 是当前最佳监督 loss，因此只对 W0 做最终校准。所有校准器均仅在 inner-validation 拟合，参数冻结后才在 outer-test 评价：

| 任务 / 方法 | Outer Brier | Outer ECE | 说明 |
|---|---:|---:|---|
| ACS 未校准 | 0.517459 | 0.023840 | 四分类 softmax |
| ACS Temperature Scaling | **0.516842** | **0.022333** | `T=1.123866` |
| OMI 未校准 | 0.046773 | 0.016618 | 独立 sigmoid |
| OMI Platt | **0.046358** | **0.010304** | 三种 OMI 校准中最佳 |
| OMI Isotonic | 0.046391 | 0.013323 | 无参数阈值泄漏 |
| OMI Beta calibration | 0.046378 | 0.012417 | 低于未校准 ECE |

ACS Temperature Scaling 与 OMI 三种校准均未使用 outer-test 拟合。OMI 已报告 0.5、max-F1、Youden、Sensitivity ≥ 0.90、Specificity ≥ 0.90 工作点；阈值全部来自 inner-validation。校准只改变概率尺度，不改变 ACS argmax，因此 ACS Accuracy、Macro F1 与混淆矩阵保持不变。

为判断 joint learning 是否存在正迁移或负迁移，使用相同预训练 checkpoint、LoRA 配置、nested Fold0、seed、optimizer、scheduler、batch size 和 preprocessing 完成 `ACS-only` 与 `OMI-only`。Joint 对照直接复用 W0。

| 任务 | 方法 | Primary AUPRC | AUROC | F1 | Balanced Accuracy | 关键少数类 / 子组 |
|---|---|---:|---:|---:|---:|---|
| ACS | Joint W0 | **0.546685** | **0.783155** | **0.525076** | **0.516938** | NSTEMI AUPRC 0.287459 |
| ACS | ACS-only W0 | 0.539538 | 0.777611 | 0.483992 | 0.492823 | NSTEMI AUPRC 0.270281 |
| OMI | Joint W0 | **0.412660** | **0.897364** | **0.367113** | **0.823288** | NSTE-ACS 内 AUPRC 0.227752 |
| OMI | OMI-only W0 | 0.358639 | 0.884419 | 0.340234 | 0.806721 | NSTE-ACS 内 AUPRC 0.179534 |

Joint − single 的 ACS Macro AUPRC 为 `+0.007147`、NSTEMI AUPRC 为 `+0.017178`、OMI AUPRC 为 `+0.054021`。Fold0 中两个任务均为正迁移，未发现 multi-task interference；OMI 的联合收益更明显。OMI 两组的 inner-validation Youden 阈值均为 0.04，因此差值不是由不同阈值单独造成。

训练成本：Joint W0 为 5,236 s；ACS-only 为 5,465 s；OMI-only 为 4,601 s。ACS-only 运行后段与另一用户的约 7.2 GiB GPU0 进程重叠，物理卡峰值 38,369 MiB 不可归因于本模型；共享发生前本任务观测约 31,078 MiB。OMI-only 峰值 31,546 MiB，Joint W0 为 31,541 MiB。三组都出现训练 loss 继续下降而验证 loss 回升，early stopping 分别保留 Joint epoch 5、ACS-only epoch 5、OMI-only epoch 3 checkpoint。

上述是 Fold0 screening evidence，不是五折均值或显著性结论。若要把 joint learning 作为论文贡献，需要完整 outer 5-fold 复核。

## 10. 最终问题回答

1. 评估问题是否完全修复：Phase 1 已通过；nested evaluator 已通过 smoke gate。
2. 静态类别加权是否有效：当前 Fold0 无效；W1–W3 三个 primary AUPRC 均低于 W0，因此未进入完整五折。
3. RDR 是否优于 unweighted：否，W4–W6 三个 primary AUPRC 均低于 W0。
4. RDR 是否真正优于 static weighting：没有可信证据；虽然部分数值高于 W3，但最终 checkpoint 来自未体现 RDR 收益的 warmup 状态，不能归因于 adaptive weighting。
5. NSTEMI 是否稳定改善：否；Fold0 AUPRC 从 W0 的 0.287459 降为 0.263176。
6. OMI 是否稳定改善：否；Fold0 AUPRC 从 0.412660 降为 0.388518。
7. NSTEMI vs UA 是否改善：未观察到足以通过 primary gate 的整体改善。
8. OMI among NSTE-ACS 是否改善：未观察到足以通过 primary gate 的整体改善。
9. calibration 是否恶化：RDR 相对 W0 轻微恶化（ACS Brier 0.517459→0.519608，OMI 0.046773→0.048415）；对最终 W0 做的 Temperature/Platt 校准则改善到 ACS Brier 0.516842、OMI Brier 0.046358。
10. RDR 时间和显存开销：profiling 平均约 28.45 ms/batch；无可测的额外 allocated-memory 峰值，但训练总时长未换来更优 checkpoint。
11. 是否值得作为论文 contribution：不适合作为正向性能创新点；可作为严格的负结果/消融。
12. 是否需要重新考虑 RL：当前仍为 **NO-GO**。校准已改善概率质量，Joint W0 又在 Fold0 同时优于两个 single-task，现有剩余问题仍可由监督学习、校准与任务设计研究，不构成 RL 的科学必要性。

## 11. 当前判定

Adaptive-weighting final decision：**NO-GO**。当前结果只能说明本研究中的 inverse-frequency static weighting 与该 RDR 适配没有获益，不能宣称所有类别不平衡方法都无效。

RL：**NO-GO（本轮禁止且未实现）**。

## 12. 最终终端摘要

Evaluation repair:
- status: PASS
- provenance status: record-ID 显式 join；duplicate/missing prediction=0；missing label=0；patient overlap=0

Current LoRA baseline:
- ACS Macro AUPRC: 0.546685
- NSTEMI AUPRC: 0.287459
- OMI AUPRC: 0.412660

Static weighting:
- W1: ACS 0.530770 / NSTEMI 0.279219 / OMI 0.409730
- W2: ACS 0.517696 / NSTEMI 0.247335 / OMI 0.381572
- W3: ACS 0.495348 / NSTEMI 0.240624 / OMI 0.361160

Adaptive weighting:
- W4: ACS 0.534947 / NSTEMI 0.263176 / OMI 0.388518
- W5: ACS 0.534947 / NSTEMI 0.263176 / OMI 0.388518
- W6: ACS 0.534947 / NSTEMI 0.263176 / OMI 0.388518

W6 vs W0:
- Δ ACS Macro AUPRC: -0.011738
- Δ NSTEMI AUPRC: -0.024283
- Δ OMI AUPRC: -0.024142
- 95% CI: 未执行；Fold0 gate 未通过，按预注册方案停止完整五折与 paired bootstrap

W6 vs W3:
- Δ ACS Macro AUPRC: +0.039598
- Δ NSTEMI AUPRC: +0.022552
- Δ OMI AUPRC: +0.027358
- 95% CI: 未执行；且 W4/W5/W6 最终 checkpoint/predictions 相同，来自 warmup checkpoint，不能把该差值归因于 RDR

Calibration:
- W0: ACS Brier/ECE 0.517459/0.023840；OMI 0.046773/0.016618
- W3: ACS Brier 0.557561；OMI Brier 0.085517（未进入最终校准）
- W6: ACS Brier 0.519608；OMI Brier 0.048415（未进入最终校准）
- W0 calibrated: ACS Temperature Brier/ECE 0.516842/0.022333；OMI Platt 0.046358/0.010304

RDR efficiency overhead:
- time: canonical profiling 28.45 ms/batch；W6 wall time 4,740 s、W0 5,236 s，但 early-stopping 轨迹不同，wall time 差不可解释为负开销
- VRAM: 未观察到额外物理峰值（W6 31,129 MiB，W0 31,541 MiB）；未保存独立的 `max_memory_allocated` 因此不声称显存节省

Adaptive-weighting final decision:
NO-GO

Is RL scientifically justified now?
NO

Recommended next step:
保留 W0 LoRA joint 作为当前工程方案；若要把 joint positive transfer 写成论文贡献，先扩展 ACS-only / OMI-only / Joint 至完整 outer 5-fold，而不是进入 RL。
