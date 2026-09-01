# ECG-LFM RL Post-training 研究执行报告

> 本报告按 `ECG_LFM_RL_PostTraining_Conditional_GO_Codex_Report.md` 执行并持续更新。当前仍为 **RL NO-GO**；本阶段只补齐进入 RL 前的监督证据，不实现或运行 RL。

## 1. 当前任务与工程基线

- ACS：互斥四分类 `OTHER / STEMI / NSTEMI / UA`，4 logits、softmax、CrossEntropyLoss；
- OMI：独立二分类 `OMI / NOMI`，1 logit、sigmoid、BCEWithLogitsLoss；
- 联合损失：`L_ACS + 1.0 × L_OMI`；
- 当前工程基线：ECG-LFM LoRA，rank 8、alpha 16、dropout 0.05；
- 外层：既有 patient-level 5-fold；
- 内层：每个 outer-train 内固定的 patient-level inner-train / inner-validation；
- inner-validation 用于 early stopping、checkpoint、threshold 与 calibration；outer-test 每个方法只评价一次。

## 2. Phase 0 审计

状态：**完成**。

- 本地与远端关键 runner/config SHA-256 已交叉核对；
- Fold0–4 nested manifest、patient assignment 与 split provenance 均存在；
- Fold0 的 Joint W0、ACS-only W0、OMI-only W0 checkpoint 与评估结果完整；
- 当前 W0 仍为未加权 CE + 未加权 BCE，OMI task weight 为 1.0；
- 逐例预测含患者标识，仅保留在受控服务器，不进入 Git；
- 当前没有实现 PPO、GRPO、DAPO 或其他 RL trainer。

## 3. Phase 1：Joint vs Single-task nested 5-fold

状态：**运行中（2026-09-01 启动）**。

| 任务 | Fold0 | Fold1–4 | 当前用途 |
|---|---|---|---|
| Joint W0 | 完成 | 运行中 | 检验联合学习是否稳定正迁移 |
| ACS-only W0 | 完成 | 已排队 | 与 Joint 的 ACS 指标配对比较 |
| OMI-only W0 | 完成 | 已排队 | 与 Joint 的 OMI 指标配对比较 |

两张 RTX A6000 分别绑定一个物理 GPU，在独立 `screen` 中执行；每个实验写入训练日志、runtime、显存峰值、checkpoint、inner-validation/outer-test 指标和可视化。自动 finalizer 在 15/15 组结果完成后执行：

- per-fold metrics；
- mean ± std；
- pooled outer-OOF；
- 1,000 次 patient-level paired bootstrap；
- Joint positive-transfer gate；
- `docs/ECG_LFM_joint_vs_single_nested5fold.md`。

Fold0 已有结果（仅为单折证据）：

| 任务 | 方法 | Primary AUPRC | NSTEMI / NSTE-ACS AUPRC |
|---|---|---:|---:|
| ACS | Joint W0 | 0.546685 | NSTEMI 0.287459 |
| ACS | ACS-only W0 | 0.539538 | NSTEMI 0.270281 |
| OMI | Joint W0 | 0.412660 | NSTE-ACS 0.227752 |
| OMI | OMI-only W0 | 0.358639 | NSTE-ACS 0.179534 |

这些 Fold0 差值尚不能替代完整五折和 paired CI。

## 4. Phase 2：剩余监督 Loss Fold0 screening

状态：**代码与测试完成，等待 Phase 1 gate 和 GPU；尚未启动训练**。

仅改变 loss，其他配置保持 Joint W0 不变。已实现并锁定第一轮标准参数：

| 方法 | ACS loss | OMI loss | 参数 |
|---|---|---|---|
| Focal | Focal CE | Focal BCE | gamma=2.0，不叠加 alpha/静态权重 |
| OMI-ASL | CE | Asymmetric Loss | gamma_neg=4、gamma_pos=0、clip=0.05 |
| Label smoothing | Label-smoothed CE | BCE | epsilon=0.1 |
| OMI-HNM | CE | Hard-negative BCE | 每 batch 负类 top 25%，保持负类 loss 总尺度 |

公平性约束：

- validation/checkpoint criterion 对所有候选统一为未加权 CE+BCE；
- same pretrained checkpoint、LoRA、optimizer、LR、scheduler、batch、seed、split 与 evaluator；
- 极端 logits 数值稳定性、Focal 等价边界、HNM 梯度选择与统一 validation objective 已通过服务器环境单元测试；
- outer-test 不用于修改上述参数。

## 5. 尚不能执行的阶段与原因

以下内容需要临床团队先提供并版本化，当前禁止自行编造：

- ACS 4×4 clinical cost matrix；
- OMI FN/FP clinical cost；
- PPV、Specificity、STEMI safety floor；
- ACS/OMI primary endpoint non-inferiority margin。

因此 Clinical cost-sensitive、exact 8-action direct utility、最终 supervised Pareto clinical gate 及任何 RL 阶段均未获准启动。

服务器 `/data` 在 Phase 1 启动时仅约 62 GiB 可用；12 个新增完整 ECG-LFM best checkpoint 预计约 29 GiB。所有启动器均设置磁盘安全门，低于安全余量时停止新增实验，不删除历史结果。

## 6. 当前结论

- 当前证据不能证明 RL 具有科学必要性；
- Joint positive transfer 目前只在 Fold0 出现，必须等待五折与 paired bootstrap；
- 静态权重与 RDR 的既有 Fold0 失败不能外推为 Focal、ASL、label smoothing 或 HNM 无效；
- Phase 2 完成后仍必须先构建最强 supervised baseline；
- 在临床代价与 safety constraint 未预注册前，RL 继续保持 **NO-GO**。

## 7. 可追溯实现

- Joint/Single 五折队列：`scripts/run_joint_vs_single_nested_fold.sh`；
- 五折自动汇总：`experiments/ecg_lfm/summarize_joint_vs_single_nested5fold.py`；
- 自动收尾：`scripts/finalize_joint_vs_single_nested5fold.sh`；
- 监督筛选 criterion：`third_party/ECG-LFM/fairseq_signals/criterions/acs_joint_supervised.py`；
- 监督筛选 runner：`experiments/ecg_lfm/run_supervised_screen_fold.py`；
- 安全启动器：`scripts/run_supervised_screen_fold0.sh`；
- 单元测试：`tests/test_joint_vs_single_nested5fold_summary.py`、`tests/test_supervised_screening_losses.py`。
