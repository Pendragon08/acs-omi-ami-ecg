# ECG-LFM Cross-task Clinical RL v2 实验结果报告

> 执行日期：2026-09-01
> 执行依据：`ECG_LFM_CrossTask_NSTE_OMI_RL_v2_Codex_Report.md`
> 实验状态：v2A Head-only Fold0 网格已完成；唯一候选已冻结并完成一次 outer Fold0 评估；未启动 v2B、完整五折或多随机种子。
> 重要边界：本报告的 `outer Fold0` 是训练池内预先定义的 outer fold，不是 1,995 条外部平台/隔离测试集；外部平台测试本轮没有访问。

## 1. 执行范围与最终结论

本轮不是重新设计 ECG-LFM，也没有实现 PPO、GRPO、DAPO 或其他 RL trainer。实验在现有 corrected ACS+OMI 联合任务和 W0 checkpoint 上，仅验证报告提出的 **Cross-task Clinical RL v2A Head-only**：

* 保持 ACS 四分类与 OMI 独立二分类的因子化输出；
* 将 `NSTE-ACS + OMI` 的临床交叉状态写入 reward；
* 使用短 update budget、adaptive target-KL、hard KL stop/rollback；
* 使用小系数 ACS/OMI pairwise ranking preservation；
* 用 inner-validation 选择 checkpoint、阈值和候选，不看 outer-test；
* 只有冻结唯一候选后，才进行一次 outer Fold0 inference 和 calibration。

结论为 **CONDITIONAL（条件性继续）**，不是已经证实有效的 GO：

1. v2A 四个候选中，两个 `target_KL=0.01` 候选通过全部 Safety Gate，两个 `target_KL=0.005` 候选触发连续 hard-KL stop 并 rollback。
2. 选中的 `LR=1e-6, target_KL=0.01, ranking_coefficient=0.01` 在 inner-validation 的 clinical utility 从 1.044858 增至 1.057250，但 ACS Macro AUPRC 和 OMI AUPRC 分别轻微下降 0.000294 和 0.000645；收益主要来自 OMI F1、OMI PPV 和 NSTE-OMI AUPRC。
3. 一次性 outer Fold0 中，ACS Macro AUPRC、OMI AUPRC 和 NSTE-OMI AUPRC 仅有极小正变化；OMI F1、Specificity、PPV 轻微下降，不能称为稳定涨点。
4. 未完成五折、多 seed 和 paired bootstrap，因此目前只能把它作为安全的探索性候选，不能作为论文中已验证的 RL 方法创新。

## 2. 当前任务定义与数据

实际运行代码、manifest 和 checkpoint 位于服务器：

```text
/data/code/ouyanghao/oyh/ACS-OMI-AMI
```

用户环境说明中还出现了 `/data/code/ouyanghao/oy/ACS-OMI-AMI`。该路径实际没有本轮 W0/data/manifest 完整工程；因此本轮根据真实文件和该目录下 `AGENTS.md` 使用 `/oyh/ACS-OMI-AMI`，没有把路径差异隐藏。

任务严格为两个 head：

| Head | 标签 | 输出 | 概率/决策 |
|---|---|---:|---|
| ACS | `OTHER / STEMI / NSTEMI / UA`，互斥 | 4 logits | `softmax(logits[:, :4])`，`argmax` |
| OMI | `NOMI / OMI`，独立二分类 | 1 logit | `sigmoid(logits[:, 4])`，inner-validation 冻结阈值 |

因此 policy 是：

```text
Categorical(ACS-4) × Bernoulli(OMI)
```

而不是 8 类 softmax，也不是四个独立 ACS sigmoid。

数据和划分的真实证据：

* 训练池 17,960 条 ECG；隔离/外部平台数据 1,995 条；master 总数 19,955。
* 外层为 patient-level 5-fold，seed=2026；本轮只使用 outer Fold0 的 nested split。
* Fold0 inner-validation 为 2,861 条记录、2,723 名患者；outer-test 为 3,606 条记录。
* split provenance SHA-256：`4ef0b33713fabcb8b08d426220dbe3ba9905710b21f5a5cfed092bf37305be10`。
* outer Fold0 对齐审计：duplicate record ID=0、missing label=0、missing prediction=0、label mismatch=0、patient mismatch=0；三组 split 的 patient overlap 全为 0。

## 3. W0 监督参考基线

W0 来源：

```text
results/adaptive_weighting/w0/fold0/checkpoint_best.pt
```

checkpoint SHA-256：

```text
ca020660b95e226cf698d05e4abe1e8639f43bcf7068c94cd13eaebf1915433c
```

真实结构为 ECG-LFM pretrained encoder、1024-D pooled representation、LoRA rank=8/alpha=16/dropout=0.05 以及 `Linear(1024, 5)` joint head。W0 的 base encoder frozen，LoRA 和 head 是监督微调部分；本轮 v2A 从 W0 重新加载后只训练 head，LoRA 与 base 都冻结。

W0 的 loss 和优化配置保持不变：

```text
ACS: CrossEntropyLoss（acs_weight=null）
OMI: BCEWithLogitsLoss（omi_pos_weight=null）
joint loss: L_ACS + 1.0 × L_OMI
optimizer: Adam，betas=(0.9, 0.98)，weight_decay=0
learning rate: 1e-4（W0 监督训练配置）
batch size: 64，gradient accumulation=4（有效 batch=256）
scheduler: step，decay_period=112，decay=0.99
```

W0 本轮 RL runner 使用同一 inner-validation 的 reference bundle（OMI Youden 阈值 0.04）：

| 指标 | W0 inner-validation |
|---|---:|
| ACS Macro AUROC | 0.798901 |
| ACS Macro AUPRC | 0.558378 |
| ACS Macro F1 | 0.531053 |
| NSTEMI AUPRC | 0.313890 |
| NSTEMI F1 | 0.324503 |
| OMI AUROC | 0.918423 |
| OMI AUPRC | 0.466406 |
| OMI F1 | 0.409524 |
| OMI Sensitivity | 0.891192 |
| OMI Specificity | 0.821964 |
| OMI PPV | 0.265842 |
| OMI among NSTE-ACS AUPRC | 0.310831 |
| ACS Brier / ECE | 0.512201 / 0.024492 |
| OMI Brier / ECE | 0.047098 / 0.017044 |
| clinical utility | 1.044858 |

utility 定义为：

```text
NSTEMI F1 + OMI F1 + OMI AUPRC among NSTE-ACS
```

## 4. 为什么转向 Cross-task RL v2

上一轮 R1-v2 已经把 RL 限制为短 update budget 和 Head-only，但没有显式表达 `NSTEMI/UA + OMI` 的联合临床状态，也没有 ranking preservation。R1-v2 的结论是：候选虽然可通过安全门，但 utility 没有超过 W0，且不能证明 AUPRC 涨点。

本轮只针对已经观察到的两个可检验问题增加机制：

* `NSTEMI + OMI` 可能比普通 NSTEMI/NOMI 更需要优先识别，因此增加显式 cross-task reward；
* R1-v2 中 hard decision 指标有所改善但整体排序指标不稳定，因此加入小系数 pairwise ranking preservation。

没有把 reward 泛化为“提高所有临床性能”，也没有根据 outer-test 调任何 reward 权重。

## 5. v2A 方法实现

### 5.1 Cross-task reward

代码：`experiments/ecg_lfm/rl/cross_task_reward.py`。

对真值为 NSTEMI+OMI 和 UA+OMI 的样本分别记录交叉状态；四种预测状态的 reward 在 YAML 中显式配置。当前运行配置为：

```yaml
reward:
  hard_scale: 0.5
  cross_task:
    enabled: true
    scale: 0.5
    nstemi_omi: {exact: 1.0, omi_only: 0.5, phenotype_only: 0.0, both_wrong: -0.5}
    ua_omi: {exact: 0.5, nstemi_omi: 0.25, phenotype_only: 0.0, both_wrong: -0.25}
```

原有 ACS/OMI hard reward 和 soft-F1 shaping 保留，cross-task reward 是增量项，不改变标签定义。

### 5.2 Adaptive target-KL 与 hard stop

代码：`experiments/ecg_lfm/rl/adaptive_kl_controller.py`。

* max updates=100；每 25 updates 在 inner-validation 评估。
* target-KL 网格为 0.005、0.01。
* KL coefficient 根据 batch KL 与 target 的比例在 `[1e-4, 1.0]` 内自适应调整。
* `hard_kl_limit = 4 × target_KL`；连续两次越界则停止并回滚到最后一个安全状态或 W0 head。
* 选中候选的 mean factorized KL=0.000720，低于 0.05 的 inner safety limit；没有发生 rollback。

### 5.3 Ranking preservation

代码：`experiments/ecg_lfm/rl/ranking_preservation.py`。

对 OMI positive/negative 以及 ACS 四个 one-vs-rest 类别采样正负 pair，使用 `softplus(-(s_positive-s_negative))`，每个 positive 最多采样 2 个 negative，避免构造全 `O(B²)` pair。正式实验固定 `ranking_coefficient=0.01`；smoke 试验比较过 0.01 与 0.05，0.01 对 policy/soft-F1 的扰动较小，作为 preservation term。

### 5.4 Calibration 和 reward-hacking guard

OMI 阈值只在 W0 inner-validation 通过 Youden Index 确定为 0.04，并冻结用于候选和 outer-test；没有在 NSTEMI、UA 或 outer-test 子集单独调阈值。

每个候选都记录：

* OMI 预测阳性率在 OTHER/STEMI/NSTEMI/UA/NSTE 子集的变化；
* NSTE 子集的 OMI PPV、Specificity；
* ACS/OMI Brier 与 ECE；
* factorized KL 和 ranking loss。

## 6. 代码、配置和运行证据

本轮新增/修改的主要文件：

| 文件 | 用途 |
|---|---|
| `experiments/ecg_lfm/train_rl_v2_cross_task.py` | v2A Head-only update-budget runner |
| `experiments/ecg_lfm/rl/cross_task_reward.py` | NSTEMI/OMI、UA/OMI cross-task reward |
| `experiments/ecg_lfm/rl/ranking_preservation.py` | ACS/OMI sampled pairwise ranking |
| `experiments/ecg_lfm/rl/adaptive_kl_controller.py` | adaptive target-KL controller |
| `experiments/ecg_lfm/summarize_cross_task_rl_v2.py` | Safety-first candidate selection |
| `experiments/ecg_lfm/freeze_cross_task_v2_candidate.py` | 冻结唯一候选，禁止提前 outer-test |
| `experiments/ecg_lfm/evaluate_cross_task_v2_outer_once.py` | outer Fold0 一次性 wrapper |
| `experiments/ecg_lfm/calibrate_nested_fold.py` | 修复绝对路径执行时的 import，并完成 frozen calibration |
| `configs/ecg_lfm/rl/v2a_head_cross_task_targetkl.yaml` | 正式配置与 reward 版本 |
| `scripts/run_ecg_lfm_v2a_cross_task_head.sh` | 两 GPU lane 的严格单 GPU 启动脚本 |
| `tests/test_ecg_lfm_clinical_rl.py` | cross reward、ranking、KL、outer guard 单元测试 |

远端执行使用 `/data/code/ouyanghao/miniconda3/envs/ecg-lfm/bin/python`，W&B disabled；训练进程按要求在 screen 中绑定单张 GPU。GPU0/GPU1 均只承担各自 lane，未使用外部平台标签。

## 7. 测试与 smoke

远端结果：

```text
python tests/test_ecg_lfm_clinical_rl.py
20/20 tests PASS
```

smoke 阶段在 GPU0/GPU1 各运行 3 updates，确认：

* factorized policy 的 ACS softmax 与 OMI sigmoid 可运行；
* cross reward 有序且无 NaN/Inf；
* sampled ranking loss 可反向传播；
* adaptive KL controller 可更新；
* outer-test guard 未被触发。

## 8. v2A 四组正式候选

四组候选均从同一个 W0 checkpoint 重新初始化，保持同一个 Fold0 nested split、seed=2026、base/LoRA 冻结、head-only、reward 配置和 ranking coefficient。

| Candidate | LR | target KL | updates/status | Safety | clinical utility | ACS Macro AUPRC | NSTEMI AUPRC | NSTEMI F1 | OMI AUPRC | OMI F1 | NSTE OMI AUPRC | mean KL | time (s) | peak VRAM (GiB) |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R2A-1 | 1e-6 | 0.005 | 7 / hard stop + rollback | FAIL | – | – | – | – | – | – | – | – | 287.70 | 4.889 |
| R2A-2 | 3e-6 | 0.005 | 7 / hard stop + rollback | FAIL | – | – | – | – | – | – | – | – | 266.64 | 4.889 |
| R2A-3 | 1e-6 | 0.010 | 100 / complete | PASS | 1.057250 | 0.558084 | 0.316270 | 0.323024 | 0.465760 | 0.421702 | 0.312525 | 0.000720 | 507.90 | 4.890 |
| R2A-4 | 3e-6 | 0.010 | 100 / complete | PASS | 1.056510 | 0.558053 | 0.316440 | 0.323232 | 0.466167 | 0.420147 | 0.313130 | 0.000682 | 458.97 | 4.890 |

`0.012?` 不是候选的最终 mean KL；两组失败候选在 update 7 因 batch KL 连续超过 `4×target` 停止，未生成可用于选择的 best checkpoint。精确的 KL 控制轨迹保存在各自 `kl_controller_history.csv`，其中 hard threshold 分别为 0.020。

按“先 Safety PASS，再最大 inner clinical utility”选择 R2A-3：

```text
run_name = R2A_cross_task_head_lr1e6_tkl010_r0p01
selected_update = 100
checkpoint_sha256 = 4235a5dd34310186e81078dc55e7174f9fee53cdac9b25c80353d74958e81c29
```

## 9. Inner-validation：W0 与选中候选

以下表格使用 runner 在相同 inner-validation 中记录的结果；OMI hard metrics 使用冻结 global threshold=0.04。

| 指标 | W0 | R2A-3 | Δ(R2A-3−W0) |
|---|---:|---:|---:|
| ACS Macro AUROC | 0.798901 | 0.798703 | -0.000199 |
| ACS Macro AUPRC | 0.558378 | 0.558084 | -0.000294 |
| ACS Macro F1 | 0.531053 | 0.531934 | +0.000881 |
| NSTEMI AUPRC | 0.313890 | 0.316270 | +0.002380 |
| NSTEMI F1 | 0.324503 | 0.323024 | -0.001479 |
| NSTEMI Sensitivity | 0.267760 | 0.256831 | -0.010929 |
| NSTEMI-vs-UA AUPRC | 0.560272 | 0.562929 | +0.002657 |
| OMI AUROC | 0.918423 | 0.918704 | +0.000281 |
| OMI AUPRC | 0.466406 | 0.465760 | -0.000645 |
| OMI F1 | 0.409524 | 0.421702 | +0.012178 |
| OMI Sensitivity | 0.891192 | 0.886010 | -0.005181 |
| OMI Specificity | 0.821964 | 0.832459 | +0.010495 |
| OMI PPV | 0.265842 | 0.276699 | +0.010857 |
| OMI among NSTE-ACS AUPRC | 0.310831 | 0.312525 | +0.001694 |
| OMI among NSTEMI AUPRC | 0.529379 | 0.526883 | -0.002496 |
| ACS Brier | 0.512201 | 0.512548 | +0.000347 |
| ACS ECE | 0.024492 | 0.032690 | +0.008198 |
| OMI Brier | 0.047098 | 0.047247 | +0.000149 |
| OMI ECE | 0.017044 | 0.018236 | +0.001192 |
| factorized mean KL | 0 | 0.000720 | +0.000720 |

解释：R2A-3 在 inner utility 上确有小幅改善，但 primary AUPRC 并没有同步提升，NSTEMI sensitivity 也下降；因此这不是“整体性能提升”。Safety Gate 使用 patient-level bootstrap standard error 和最低 0.005 margin，R2A-3 通过的是 non-inferiority-style safety，不是统计显著性检验。

## 10. Safety Gate 与 reward-hacking 审计

R2A-3 的八个安全检查均为 true：

```text
acs_macro_auprc_guard
omi_auprc_guard
omi_ppv_guard
omi_specificity_guard
stemi_f1_guard
acs_brier_guard
omi_brier_guard
kl_guard
```

R2A-3 的 NSTE-ACS reward-hacking 指标：

| 指标 | W0 reference | R2A-3 | 变化 |
|---|---:|---:|---:|
| NSTE pred OMI rate | 0.210615 | 0.197136 | ratio=0.936 |
| NSTE OMI PPV | 0.212000 | 0.226496 | +0.014496 |
| NSTE OMI Specificity | 0.823950 | 0.838248 | +0.014298 |
| global pred OMI rate | 0.226145 | 0.216008 | -0.010136 |

没有出现“为了提高 sensitivity 而把大量 NSTE 预测为 OMI”的 rate explosion 或 PPV collapse，reward-hacking guard=PASS。STEMI 子集的 OMI 预测阳性率仍然较高，因此不能把 guard PASS 解读为临床校准已经完美。

## 11. 一次性 outer Fold0 评估

选择规则、checkpoint 和阈值冻结后，才运行：

```text
results/ecg_lfm_cross_task_rl_v2_evaluation/
  R2A_cross_task_head_lr1e6_tkl010_r0p01/fold0/
```

provenance 结果：

* `outer_test_used_for_fit=false`；calibration 只在 inner-validation 拟合。
* `outer_test_evaluations=1`；冻结文件状态为 `outer_test_evaluated_once`。
* record-ID join、预测覆盖和 patient overlap 审计全部 PASS。
* outer evaluator 没有用于重新选择 LR、target KL、reward、ranking coefficient 或 threshold。

### 11.1 W0 与 R2A-3 的 outer Fold0 核心指标

| 指标 | W0 | R2A-3 | Δ |
|---|---:|---:|---:|
| ACS Macro AUROC | 0.783155 | 0.783145 | -0.000010 |
| ACS Macro AUPRC | 0.546685 | 0.546754 | +0.000070 |
| ACS Macro F1 | 0.525076 | 0.525979 | +0.000903 |
| ACS Balanced Accuracy | 0.516938 | 0.517495 | +0.000557 |
| ACS Accuracy | 0.587077 | 0.586800 | -0.000277 |
| OMI AUROC | 0.897364 | 0.897379 | +0.000015 |
| OMI AUPRC | 0.412660 | 0.412816 | +0.000156 |
| OMI F1（Youden, thr=0.04） | 0.367113 | 0.365366 | -0.001746 |
| OMI Sensitivity | 0.831169 | 0.831169 | 0.000000 |
| OMI Specificity | 0.815407 | 0.813926 | -0.001481 |
| OMI PPV | 0.235583 | 0.234146 | -0.001436 |
| OMI NPV | 0.986027 | 0.986001 | -0.000025 |
| OMI among NSTE-ACS AUPRC | 0.227752 | 0.227780 | +0.000027 |
| NSTEMI-vs-UA AUPRC | 0.526039 | 0.526040 | +0.000001 |

outer Fold0 共有 3,606 条记录。R2A-3 的 ACS Macro AUPRC/OMI AUPRC 只增加约 `7e-5/1.6e-4`，远小于可以据此宣称方法有效的证据强度；临床 utility（用 outer 结果描述性计算）由 W0 的 0.901359 变为 R2A-3 的 0.900438，反而下降 0.000921。

### 11.2 Outer ACS 逐类指标

| 类别 | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV | NPV |
|---|---:|---:|---:|---:|---:|---:|---:|
| OTHER | 0.710196 | 0.701089 | 0.675730 | 0.735746 | 0.547699 | 0.624767 | 0.669410 |
| STEMI | 0.949454 | 0.705526 | 0.662139 | 0.677083 | 0.968053 | 0.647841 | 0.971861 |
| NSTEMI | 0.797139 | 0.287524 | 0.307292 | 0.240816 | 0.976198 | 0.424460 | 0.946351 |
| UA | 0.675791 | 0.492878 | 0.458756 | 0.416333 | 0.788714 | 0.510806 | 0.718315 |

### 11.3 Outer OMI operating points

OMI threshold 均由 inner-validation 选出，outer 只应用：

| 阈值策略 | threshold | AUROC | AUPRC | Sensitivity | Specificity | PPV | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed 0.5 | 0.50 | 0.897379 | 0.412816 | 0.255411 | 0.984889 | 0.536364 | 0.346041 |
| max-F1 | 0.17 | 0.897379 | 0.412816 | 0.610390 | 0.925926 | 0.360614 | 0.453376 |
| Youden J | 0.04 | 0.897379 | 0.412816 | 0.831169 | 0.813926 | 0.234146 | 0.365366 |
| Sensitivity≥0.90 | 0.03 | 0.897379 | 0.412816 | 0.878788 | 0.780444 | 0.215042 | 0.345532 |
| Sensitivity≥0.95 | 0.01 | 0.897379 | 0.412816 | 0.930736 | 0.618370 | 0.143047 | 0.247982 |
| Specificity≥0.90 | 0.10 | 0.897379 | 0.412816 | 0.722944 | 0.893037 | 0.316288 | 0.440053 |

这张表说明 threshold 仍然是临床工作点的重要因素；不能只看 AUROC，也不能把 Youden 阈值结果当作所有场景的唯一最优决策。

## 12. Calibration

### 12.1 未校准与 inner-fitted calibration 的 outer 结果

ACS 使用 temperature scaling，OMI 分别比较 Platt、Beta、Isotonic；所有变换只在 inner-validation fit，然后冻结到 outer。

| 方法 | ACS temperature | ACS Brier（uncal / scaled） | ACS ECE（uncal / scaled） | OMI 最佳变换 | OMI Brier（uncal / calibrated） | OMI ECE（uncal / calibrated） |
|---|---:|---:|---:|---|---:|---:|
| W0 | 1.123866 | 0.517459 / 0.516842 | 0.023840 / 0.022333 | Platt | 0.046773 / 0.046358 | 0.016618 / 0.010304 |
| R2A-3 | 1.124318 | 0.517623 / 0.517011 | 0.022925 / 0.018856 | Platt | 0.046796 / 0.046354 | 0.016311 / 0.010036 |

R2A-3 的 ACS calibrated Brier 比 W0 高约 0.000169，但 ECE 更低；OMI Platt Brier/ECE 略优于 W0。整体没有出现 R1-v2 那样明显的 calibration 崩溃，但也没有证据表明 calibration 得到具有统计意义的改善。

## 13. 训练稳定性与资源成本

| 项目 | R2A-3 |
|---|---:|
| trainable scope | joint head only |
| max updates | 100 |
| validation checkpoints | update 25/50/75/100 |
| wall time | 507.90 s（约 8.5 min） |
| peak allocated GPU memory | 5,250,619,904 bytes（约 4.890 GiB） |
| outer inference time | 130.88 s（含 inference，不含 calibration） |
| hard stop/rollback | 未触发 |
| final factorized mean KL | 0.000720 |

两个 `target_KL=0.005` 候选分别在 287.70 s 和 266.64 s 后于 update 7 停止；这验证 hard KL stop 能够阻止不稳定候选继续消耗计算资源。所有训练进程结束后 screen 已清空，GPU0/GPU1 无本轮残留训练进程。

## 14. 已知评估差异与限制

候选训练器在 inner-validation 记录时使用 batch size=32、num_workers=0；统一 `evaluate_nested_fold.py` 的一次性 outer wrapper 使用 batch=64、workers=8。由于 ECG batch padding/混合精度的批次形状不同，统一 evaluator 重算的 R2A-3 inner metrics 与训练器记录会出现约 `1e-4~4e-4` 的小差异。outer W0 与 R2A-3 使用同一个 evaluator、同一个 batch=64，因此 outer 对比仍然是同 evaluator 下的公平描述性比较。

这不是重新访问 outer-test 的理由；本轮已遵守 outer 只访问一次。后续若要做正式确认性实验，应把训练时的 validation loader 与最终 evaluator 的 batch/precision 统一，并重新预注册评估协议。

本轮明确没有：

* v2B LoRA+Head low-LR study（因为 v2A 并非四组全失败，且已有两个安全候选）；
* 5-fold cross-task RL；
* seed=2027/2028 或更多多 seed；
* W6-W0 等 paired bootstrap CI；
* 1,995 条外部平台测试；
* 任何 RL 后再调参。

## 15. 是否达到 v2A GO 条件

报告定义的 Fold0 GO 需要：ACS Macro AUPRC 与 OMI AUPRC non-inferior、calibration non-inferior、KL safe，并且至少一个 NSTEMI sensitivity/F1、OMI among NSTEMI 或 OMI among NSTE-ACS 指标有明确改善。

本轮 inner-validation 通过了 safety-style non-inferiority gate，且 NSTE-OMI AUPRC、OMI F1/PPV 有改善；但 outer Fold0 的 hard OMI F1/PPV/Specificity 没有改善，primary AUPRC 变化接近零，且没有多折/多 seed 证据。因此严格判定为：

```text
Fold0 exploratory status: CONDITIONAL
论文级 efficacy claim: NO
```

候选可以保留为后续完整验证的起点，但不能写成“Cross-task RL 已经提高 ECG-LFM 性能”。

## 16. 是否应该继续 v2B 或 RL

当前最合理的后续顺序不是直接扩大 RL：

1. 先统一 runner/evaluator 的 validation batching 和 precision，避免小数值差异；
2. 在预注册的同一 nested split 上完成 v2A 的完整 5-fold 和至少 3 个 seed；
3. 对 ACS Macro AUPRC、NSTEMI AUPRC、OMI AUPRC、OMI F1/PPV、NSTE-OMI AUPRC 做 patient-level paired bootstrap；
4. 如果 cross-task reward 的少数类收益在多折多 seed 保持、primary AUPRC non-inferior 且 calibration 不恶化，再考虑 v2B LoRA+Head；
5. 如果收益仍只存在于 thresholded F1 而排序 AUPRC 不提升，应优先回到 supervised cost-sensitive loss、calibration 或 threshold policy，而不是继续增加 RL 自由度。

## 17. 文件和可追溯性索引

本地仓库保存了不含 checkpoint、原始 ECG、patient-level prediction CSV 和 raw JSONL 的可审计结果：

```text
results/ecg_lfm_cross_task_rl_v2/head_only_selection/candidates.csv
results/ecg_lfm_cross_task_rl_v2/head_only_selection/selection.json
results/ecg_lfm_cross_task_rl_v2/R2A_cross_task_head_*/fold0/
results/ecg_lfm_cross_task_rl_v2_evaluation/R2A_cross_task_head_lr1e6_tkl010_r0p01/fold0/
```

每个 formal candidate 目录包括（若该候选生成）：

* `training_summary.json`
* `launch_provenance.json`
* `reference_safety_guard.json`
* `cross_reward_history.csv`
* `ranking_history.csv`
* `kl_controller_history.csv`
* `inner_validation_update*.json`
* `rl_training_curves.png`、`rl_inner_validation_curves.png`

冻结/outer 文件包括：

* `frozen_config.json`
* `freeze_manifest.json`
* `outer_evaluation_started.json`
* `metrics.json`
* `calibration.json`
* `provenance_alignment_report.json`
* `confusion_matrices.png`
* `roc_pr_curves.png`
* `reliability_diagrams.png`

服务器保留完整 `train_log.jsonl` 和 checkpoint，未提交 GitHub，符合项目的数据与模型材料保护约束；本地报告和 curated CSV/JSON/PNG 足以复核候选选择、KL 轨迹、reward 变化、校准和 outer 结果。

## 18. 最终摘要

```text
Current best: W0 remains the strongest established engineering reference; R2A-3 is the only frozen cross-task RL v2 candidate.
Main bottleneck: RL v2 did not produce a consistent ranking-level gain; hard OMI operating-point metrics and NSTEMI sensitivity remain fragile.
Evidence: two target-KL=0.01 candidates passed safety; target-KL=0.005 candidates hard-stopped; R2A-3 inner utility +0.012392 but inner ACS/OMI AUPRC slightly down; outer changes are near zero.
Can supervised learning still solve it?: Yes, threshold/calibration, class-imbalance and cost-sensitive supervised baselines remain easier to interpret and are not exhausted.
Why RL may/may not be necessary: cross-task asymmetric utility is scientifically meaningful, but current one-fold/one-seed evidence does not show that CE+BCE plus calibration cannot address it.
Recommended next experiment: unify evaluator batching, then run pre-registered v2A full 5-fold × multi-seed paired analysis; do not tune on outer-test.
Final decision: CONDITIONAL (exploratory RL candidate, no paper-level efficacy claim)
```
