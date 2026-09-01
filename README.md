# ACS 四状态与 OMI 联合 ECG 分类

> **最新版实验总报告（2026-09-01）：**
> [docs/reports/实验结果总报告_最终.md](docs/reports/实验结果总报告_最终.md)
>
> 报告已合并修正五折重评估、严格 nested Fold0 W0–W6、W0 校准、
> ACS-only / OMI-only / Joint 对照和外部平台历史结果。当前推荐方案为
> **ECG-LFM LoRA Joint W0（未加权 CE + 未加权 BCE）**。

本仓库保存 ACS/OMI ECG 研究中可公开版本化的代码、配置、聚合实验结果、脱敏训练曲线和中文报告。原始 ECG、患者级表格、逐记录预测、下载的预训练权重和训练 checkpoint 不进入 GitHub。

## 当前任务

当前模型是共享 ECG-LFM encoder 的双任务模型，不是四个独立 sigmoid 标签：

- **ACS Head**：`OTHER / STEMI / NSTEMI / UA` 四个互斥状态，输出 4 logits，使用 softmax 与 CrossEntropyLoss；
- **OMI Head**：独立的 `OMI / NOMI` 二分类，输出 1 logit，使用 sigmoid 与 BCEWithLogitsLoss；
- 联合损失：`L = L_ACS + 1.0 × L_OMI`。

`OTHER` 表示明确的 non-ACS 状态。`[0,0,0]` 的 non-ACS 记录不会被错误删除；OMI 与 ACS 状态独立存在。

## 数据与划分边界

- 当前训练池：17,960 条 ECG；
- 隔离测试集：1,995 条 ECG；
- outer split：既有 patient-level 5-fold，seed 2026；
- 本轮 loss 实验：在每个 outer-train 内新增 patient-level inner-train / inner-validation；
- inner-validation 只用于 early stopping、checkpoint selection、校准和阈值选择；
- outer-test 只在模型与阈值冻结后评价一次；
- 所有 split 均保存聚合 provenance 与 SHA-256，患者级 assignment 不上传。

## 当前工程基线

当前推荐的工程基线为 ECG-LFM LoRA：

- rank 8；alpha 16；dropout 0.05；
- 学习率 `1e-4`；Adam `(0.9, 0.98)`；
- batch size 64，gradient accumulation 4，有效 batch size 256；
- 原始 encoder 参数冻结；训练 LoRA 参数与联合分类头；
- checkpoint criterion：所有 loss 方法统一使用未加权 inner-validation CE+BCE。

## 评估修复

本轮已完成以下修复：

- `get_logits(normalize=True)` 不再对全部 5 logits 做 softmax；
- ACS 使用 `softmax(logits[:, :4])`，OMI 使用 `sigmoid(logits[:, 4])`；
- prediction 与标签通过 `record_id` 显式 join；
- duplicate/missing prediction、missing label、patient overlap 均作为硬门控；
- 新增 Conditional ACS、NSTEMI vs UA、OMI among NSTE-ACS 与校准评价；
- 新增 15 项概率语义、对齐、静态权重、RDR 与单任务对照单元测试。

详见 [修正评估报告](docs/ECG_LFM_corrected_evaluation_report.md)。

## 自适应权重 Fold0 实验

所有 W0–W6 使用相同 ECG-LFM LoRA、相同 nested Fold0 和相同训练设置，只改变训练阶段 loss weighting。

| 方法 | ACS loss | OMI loss | ACS Macro AUPRC | NSTEMI AUPRC | OMI AUPRC |
|---|---|---|---:|---:|---:|
| W0 | Unweighted CE | Unweighted BCE | **0.546685** | **0.287459** | **0.412660** |
| W1 | Static weighted CE | Unweighted BCE | 0.530770 | 0.279219 | 0.409730 |
| W2 | Unweighted CE | Static pos-weight BCE | 0.517696 | 0.247335 | 0.381572 |
| W3 | Static weighted CE | Static pos-weight BCE | 0.495348 | 0.240624 | 0.361160 |
| W4 | RDR-CE | Unweighted BCE | 0.534947 | 0.263176 | 0.388518 |
| W5 | Unweighted CE | RDR-BCE | 0.534947 | 0.263176 | 0.388518 |
| W6 | RDR-CE | RDR-BCE | 0.534947 | 0.263176 | 0.388518 |

RDR 基于 NeurIPS 2024 官方仓库 `GoodMorningPeter/RDR`，固定 commit `bb55ea9f906f00a2d8938a97543fc232b5d26645`。canonical pooled 1024-D profiling 无 NaN/Inf；warmup 仅依据 W0 inner-validation 冻结为 363 updates。

W4/W5/W6 的最终 inner/outer predictions 完全一致：RDR 在 warmup 后确实启用，但没有刷新 warmup 阶段的最佳 checkpoint。三个 primary AUPRC 均低于 W0，因此按预注册 gate 判定为 **NO-GO**，没有启动完整五折、多 seed 或 paired bootstrap。该结论只针对当前静态权重和 RDR 适配，不能泛化为所有类别不平衡方法均无效。

完整分析见 [自适应权重实验报告](docs/ECG_LFM_adaptive_weighting_experiment_report.md)。

## R1-v2 RL 稳定化研究

R1-v2 在同一 W0 与 nested Fold0 上将 RL 改为 100-update budget，25 updates 验证一次，并加入 adaptive target-KL、hard KL stop 与 rollback。四组 Head-only（LR=1e-6/3e-6 × target-KL=0.005/0.01）均完成且 Safety PASS，但最高 clinical utility 1.039309 低于 W0 的 1.044860，ACS/OMI primary AUPRC 也未提升；因此本轮不进入 LoRA+Head low-LR，outer-test 未访问。

详见 [R1-v2 稳定化报告](docs/ECG_LFM_R1_v2_stabilization_report.md) 与 [R1-v2 结果索引](results/ecg_lfm_rl_v2/README.md)。候选逐 update 日志和曲线保存在 `results/ecg_lfm_rl_v2/`；checkpoint 仍只保留在服务器。

## 校准与单任务对照

当前最佳监督策略 W0 已完成严格校准。ACS Temperature Scaling 在 outer-test 将 Brier 从 0.517459 降至 0.516842、ECE 从 0.023840 降至 0.022333；OMI 的 Platt calibration 在三种候选中最佳，将 Brier 从 0.046773 降至 0.046358、ECE 从 0.016618 降至 0.010304。所有校准器只在 inner-validation 拟合。

ACS-only、OMI-only 与 Joint W0 的 nested Fold0 对照已完成。三者共享预训练模型、LoRA 配置、数据划分、随机种子、优化器、scheduler、batch 和预处理。

| 任务 | Joint W0 AUPRC | Single-task AUPRC | Joint − single |
|---|---:|---:|---:|
| ACS Macro | **0.546685** | 0.539538 | +0.007147 |
| NSTEMI | **0.287459** | 0.270281 | +0.017178 |
| OMI | **0.412660** | 0.358639 | +0.054021 |

Fold0 中两个任务方向一致地支持 joint positive transfer，尤其 OMI 获益明显；但这不是完整五折或统计显著性结论。完整指标、成本与过拟合分析见 [单任务与联合学习对照](results/single_task_vs_joint_evaluation/README.md)。

## 主要代码

- `experiments/acs_joint_task.py`：任务语义、指标、阈值、校准与图表；
- `experiments/ecg_lfm/build_nested_manifests.py`：patient-level nested split；
- `experiments/ecg_lfm/run_joint_fold.py`：W0–W6 单折训练入口；
- `experiments/ecg_lfm/evaluate_nested_fold.py`：inner-validation / outer-test 严格评价；
- `experiments/ecg_lfm/run_single_task_fold.py`：ACS-only / OMI-only 公平训练入口；
- `experiments/ecg_lfm/evaluate_single_task_nested_fold.py`：单任务 record-ID 对齐评价；
- `experiments/ecg_lfm/profile_rdr_1024d.py`：canonical RDR profiling；
- `experiments/ecg_lfm/summarize_rdr_dynamics.py`：RDR 权重轨迹；
- `third_party/ECG-LFM/fairseq_signals/modules/rdr_official.py`：官方 RDR 方程的可审计移植；
- `tests/`：概率语义、对齐、权重与 RDR 回归测试。

## 结果目录

- `results/revalidated/`：旧 Baseline / LoRA / DoRA checkpoint 的修正 pooled OOF 复核；
- `results/nested_validation/`：不含患者 ID 的 nested split 聚合 provenance；
- `results/adaptive_weighting/`：W0–W6 配置、runtime、profiling 与汇总；
- `results/adaptive_weighting_evaluation/`：聚合指标、混淆矩阵、ROC/PR、可靠性图、训练曲线和 RDR 权重图。
- `results/single_task_vs_joint_evaluation/`：ACS-only、OMI-only 与 Joint W0 的安全聚合对比。

逐记录 prediction CSV 和 checkpoint 仅保留在受控服务器，不进入本仓库。

## 安全与复现

禁止提交：

- SSH 密码或私钥；
- 原始 ECG / WFDB；
- 患者 ID、patient-level assignment 或逐记录预测；
- 下载的预训练 checkpoint 和训练 checkpoint；
- 外部测试提交文件。

数据边界见 [DATASET.md](DATASET.md)，仓库清单见 [REPOSITORY_MANIFEST.md](REPOSITORY_MANIFEST.md)。

## Cross-task Clinical RL v2

本轮按 `ECG_LFM_CrossTask_NSTE_OMI_RL_v2_Codex_Report.md` 完成了 NSTE-ACS/OMI 交叉临床奖励的 v2A Head-only Fold0 探索。两个 `target_KL=0.01` 候选通过安全门，唯一候选已冻结并完成一次 outer Fold0 评估；结果仍属于条件性探索，尚未完成五折和多 seed 确认。

详见 [Cross-task RL v2 中文实验报告](docs/ECG_LFM_CrossTask_RL_v2_report.md)、[执行日志索引](docs/logs/ECG_LFM_CrossTask_RL_v2_execution.md) 和 [结果目录](results/ecg_lfm_cross_task_rl_v2/README.md)。
