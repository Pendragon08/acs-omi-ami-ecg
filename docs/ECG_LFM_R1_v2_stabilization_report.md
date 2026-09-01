# ECG-LFM R1-v2 稳定化研究报告

> **完成状态：** Fold0 Head-only 稳定化网格已完成；四个候选均完成 100 updates、均通过 Safety PASS。尚未读取 outer-test，也未启动 LoRA+Head low-LR 阶段。

## 1. 研究目的与预注册约束

R1-v2 用于验证 update-budget、低学习率、目标 KL 控制和安全回滚能否改善上一轮 Direct RL 的训练稳定性。W0 reference、clinical reward、nested Fold0、evaluator 与 safety guards 均保持不变；本轮不修改 reward weights，不访问 outer-test。

正式候选为四组 Head-only 配置，每组都从同一 W0 checkpoint 重新初始化：

| 候选 | Head 学习率 | target-KL |
|---|---:|---:|
| H1 | 1e-6 | 0.005 |
| H2 | 1e-6 | 0.010 |
| H3 | 3e-6 | 0.005 |
| H4 | 3e-6 | 0.010 |

每个候选最多 100 个 optimizer updates，每 25 updates 做一次 inner-validation。Safety PASS 后按 clinical utility 选择，outer-test 不参与任何选择。

## 2. 当前任务与 W0 reference

- ACS Head：`OTHER / STEMI / NSTEMI / UA` 四分类，Categorical/Softmax。
- OMI Head：独立一个 logit，Bernoulli/Sigmoid；不是五路 softmax。
- 联合策略空间：`Categorical(ACS) × Bernoulli(OMI)`，共 8 个联合动作。
- W0：`results/adaptive_weighting/w0/fold0/checkpoint_best.pt`，ECG-LFM LoRA+Head 联合微调模型。
- R1-v2 第一轮：冻结 ECG-LFM encoder 与 LoRA，仅训练 `proj.weight`、`proj.bias`。
- nested Fold0：inner-train/inner-validation/outer-test patient-level 划分保持原有 split；outer-test 本轮没有加载。

H3 provenance：W0 checkpoint SHA-256=`ca020660b95e226cf698d05e4abe1e8639f43bcf7068c94cd13eaebf1915433c`；split provenance SHA-256=`4ef0b33713fabcb8b08d426220dbe3ba9905710b21f5a5cfed092bf37305be10`；配置 SHA-256=`5b09e83d2031c1360729986a9c8736be3e8e57eeaa0b665a9536d38fb2bd7724`。H1/H2/H4 的 provenance 文件分别记录其 GPU、配置 hash 和相同 W0/split hash。

W0 在同一 R1-v2 evaluator 下的 inner-validation 参考值：

| 指标 | W0 |
|---|---:|
| ACS Macro AUPRC | 0.558380 |
| NSTEMI AUPRC | 0.313890 |
| NSTEMI F1 | 0.324500 |
| OMI AUPRC | 0.466410 |
| OMI F1 | 0.409520 |
| OMI Sensitivity | 0.891192 |
| OMI Specificity | 0.821960 |
| OMI PPV | 0.265840 |
| NSTE-ACS OMI AUPRC | 0.310830 |
| ACS multiclass Brier | 0.512200 |
| OMI Brier | 0.047100 |
| Clinical utility | 1.044860 |

## 3. 代码、配置与审计证据

- 配置：`configs/ecg_lfm/rl/clinical_rl_v2_head_only.yaml`
- update-budget runner：`experiments/ecg_lfm/train_rl_v2_head_only.py`
- 候选汇总：`experiments/ecg_lfm/summarize_rl_v2_head_only.py`
- 并行启动：`scripts/run_ecg_lfm_rl_v2_head_only.sh`
- 单元测试：`tests/test_ecg_lfm_clinical_rl.py`
- 曲线：`experiments/ecg_lfm/plot_rl_training.py`

实现审计要点：

1. 固定 optimizer update 数替代 epoch；最大 100 updates。
2. 25/50/75/100 updates 均写出 inner-validation 快照。
3. KL controller 按当前 batch KL 相对 target-KL 调整系数，并限制在配置上下界。
4. KL 连续超过 `target_KL × 4` 达到 patience 时 hard stop；非有限 loss/gradient 也立即停止并回滚到最近的 Safety PASS 状态或 W0 head。
5. loader 会恢复 W0 的 LoRA `requires_grad` 标志，因此 runner 加载后再次冻结所有参数，只放开 `proj.weight` 与 `proj.bias`；Head-only 不变量已在 16 项单元测试中通过。
6. safe candidate 只在 inner-validation Safety PASS 后保存；选择器实现“Safety PASS 优先、clinical utility 最大”。

实现验证：服务器执行 `python tests/test_ecg_lfm_clinical_rl.py`，16/16 tests 通过；三个 R1-v2 脚本均通过 `py_compile`。四个正式候选均使用 `CUDA_VISIBLE_DEVICES=0` 或 `1` 的独立 screen 进程运行。

## 4. Safety 与选择规则

Safety guard 使用 W0 inner-validation 的 patient-level bootstrap margin，检查 ACS Macro AUPRC、OMI AUPRC/PPV/Specificity、STEMI F1、ACS/OMI Brier 和 mean KL。clinical utility 保持上一轮定义：

`NSTEMI F1 + OMI F1 + NSTE-ACS 中 OMI AUPRC`（权重见 YAML）。

四个候选均完成且 Safety PASS，因此没有触发“全部失败即停止”的分支；但是否进入 LoRA+Head 仍以相对 W0 的真实性能为依据。outer-test 必须等唯一 final candidate 冻结后才允许访问。

## 5. R1-v2 候选结果（Fold0 inner-validation）

结果来源：`results/ecg_lfm_rl_v2/head_only_selection/candidates.csv`、`selection.json`，以及各候选目录中的 `training_summary.json`。

| 候选 | 状态 | Safety | 选中 update | ACS Macro AUPRC | NSTEMI AUPRC | NSTEMI F1 | OMI AUPRC | OMI F1 | OMI Sens. | OMI Spec. | OMI PPV | NSTE-OMI AUPRC | mean KL | Utility | 时间(s) | 峰值显存(GiB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 1e-6/0.005 | completed | PASS | 50 | 0.558206 | 0.313800 | 0.318937 | 0.466404 | 0.409037 | 0.891192 | 0.821589 | 0.265432 | 0.310753 | 0.000069 | 1.038727 | 503.36 | 4.864 |
| H2 1e-6/0.010 | completed | PASS | 50 | 0.558206 | 0.313800 | 0.318937 | 0.466404 | 0.409037 | 0.891192 | 0.821589 | 0.265432 | 0.310753 | 0.000069 | 1.038727 | 503.34 | 4.864 |
| H3 3e-6/0.005 | completed | PASS | 25 | 0.558175 | 0.314109 | 0.320000 | 0.466397 | 0.408551 | 0.891192 | 0.821214 | 0.265023 | 0.310757 | 0.000218 | 1.039309 | 454.50 | 4.864 |
| H4 3e-6/0.010 | completed | PASS | 25 | 0.558175 | 0.314109 | 0.320000 | 0.466397 | 0.408551 | 0.891192 | 0.821214 | 0.265023 | 0.310757 | 0.000218 | 1.039309 | 454.50 | 4.864 |

H3/H4 按预注册规则并列为最高 utility（1.039309）；由于数值完全一致，选择器将 H3 作为稳定排序下的唯一 selected candidate。H1/H2 也完全一致。两种 target-KL 未产生区分，是因为实际 KL 远低于目标，controller 很快将 KL 系数降到配置下限；这说明本轮主要验证了低漂移和可回滚性，而不是 KL 目标的有效区分能力。

### 选中 H3 的完整指标

ACS 四分类：accuracy=0.588256，balanced accuracy=0.515563，macro F1=0.528992，macro AUROC=0.798837，macro AUPRC=0.558175。OTHER/STEMI/NSTEMI/UA 的 AUROC 为 0.711812/0.952162/0.846031/0.685342，AUPRC 为 0.697037/0.720890/0.314109/0.500664，F1 为 0.681394/0.676533/0.320000/0.438040。

OMI（Youden threshold=0.04）：AUROC=0.918434，AUPRC=0.466397，sensitivity=0.891192，specificity=0.821214，PPV=0.265023，NPV=0.990506，F1=0.408551，MCC=0.426662；TP/TN/FP/FN=172/2191/477/21。真实 ACS 中 NSTEMI-vs-UA AUROC/AUPRC=0.863334/0.560335；NSTE-ACS 内 OMI AUROC/AUPRC=0.877484/0.310757。校准为 ACS multiclass Brier=0.512299、OMI Brier=0.047109、OMI ECE=0.017851、OMI calibration slope/intercept=0.845667/-0.078623。

## 6. 训练过程与资源

每个候选目录均保存：

- `train_log.jsonl`：每个 update 的 loss、reward、ratio、KL、controller 系数、梯度和熵；
- `inner_validation_update25/50/75/100.json`：验证指标、Safety、utility、KL 和显存；
- `rl_training_curves.png`：优化目标、reward、KL、熵与 clipping 曲线；
- `rl_inner_validation_curves.png`：验证指标和 utility 随 update 曲线；
- `training_summary.json`：停止原因、回滚、update 数、峰值显存；
- `launch_provenance.json`：W0/checkpoint/split hash、GPU、配置和 `outer_test_loaded=false`。

四组均正常完成 update budget，未触发 hard KL stop、nonfinite rollback 或 wall-time stop。峰值显存约 4.864 GiB；单候选耗时约 454–503 s。GPU0 的总显存还包括同时运行的其他进程，但该候选自身峰值仍由 runner 记录为约 4.864 GiB。

## 7. 与 W0 及上一轮 Direct RL 的比较

H3 相对同一 runner 的 W0 变化：

| 指标 | H3-W0 |
|---|---:|
| ACS Macro AUPRC | -0.000205 |
| NSTEMI AUPRC | +0.000219 |
| NSTEMI F1 | -0.004500 |
| OMI AUPRC | -0.000013 |
| OMI F1 | -0.000969 |
| OMI Sensitivity | +0.000000 |
| OMI Specificity | -0.000746 |
| OMI PPV | -0.000817 |
| Clinical utility | -0.005551 |

H1/H2 的 ACS Macro AUPRC 为 -0.000174、utility 为 -0.006133。与上一轮 epoch-based Direct RL 相比，R1-v2 的 mean KL 仅约 6.9e-5 或 2.2e-4，显著低于旧实验的 0.16844，且没有校准灾难或漂移；但稳定性改善没有转化为性能提升。

## 8. LoRA+Head 后续决策

四个 Head-only 候选虽均 Safety PASS，但最高 utility 仍低于 W0，ACS/OMI primary AUPRC 也未提升。因此本轮不进入 LoRA+Head low-LR study，不通过继续放开 LoRA 来追逐单项指标。若未来重新研究 LoRA+Head，必须另立预注册配置、从 W0 重新初始化、沿用同一 safety gate，并在最终候选冻结前继续禁止 outer-test。

## 9. 最终结论

- 四组是否完成：是（H1–H4，均 100 updates）。
- Safety PASS：4/4。
- 最佳 Head-only 候选：H3（LR=3e-6，target-KL=0.005；H4 数值并列）。
- Head-only 是否优于 W0：否；utility、ACS Macro AUPRC、OMI AUPRC 均未改善。
- 是否进入 LoRA+Head：否，本轮停止。
- outer-test：未访问；仅生成了 inner-validation candidate checkpoint。

本报告结论只适用于 R1-v2 Fold0 inner-validation 稳定性研究，不能外推为外部测试改进，也不能据此声称 RL 优于监督学习。
