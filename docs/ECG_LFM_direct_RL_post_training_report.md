# ECG-LFM Direct RL Post-training 实验报告

> 状态：实施中。本文只记录可由配置、日志、checkpoint 与指标文件复核的结果；尚未完成的项目不填推测值。

## 1. 任务与监督起点

- ACS Head：`OTHER / STEMI / NSTEMI / UA` 四类互斥输出，Categorical/Softmax。
- OMI Head：独立单 logit 二分类输出，Bernoulli/Sigmoid。
- RL 联合策略：`Categorical(ACS) × Bernoulli(OMI)`，共 8 种联合动作；不是 8 类 softmax。
- 监督起点：`results/adaptive_weighting/w0/fold0/checkpoint_best.pt`（ECG-LFM LoRA+Head Joint W0）。
- 数据协议：现有 patient-level nested Fold0；RL 只使用 inner-train，调参、early stopping、阈值与安全门控只使用 inner-validation。
- outer-test：仅在全部配置与 checkpoint 冻结且 Fold0 safety gate 通过后评价一次。

## 2. RL 方法

本实验实现 one-step clipped contextual policy optimization。它借用 clipped probability ratio 更新，但没有序列环境、trajectory 或 critic，因此不称为标准 PPO、GRPO 或 DAPO。

Reference policy 与初始 Policy 均来自同一个 W0 checkpoint。训练前缓存 W0 在 inner-train 与 inner-validation 上的 logits，训练期间 Reference 不更新。可训练参数仅限 LoRA 矩阵和联合分类 Head；ECG-LFM 基座保持冻结。

总目标包括：采样 ACS/OMI 联合动作后的 mean-centered reward advantage、clipped policy objective、ACS/OMI Soft-F1 shaping，以及相对 W0 的精确 `KL_Categorical + KL_Bernoulli` 正则。

当前 reward 系数是预注册的 inner-validation 探索系数，不是临床专家确认的错误代价，不能作临床效用结论。

## 3. 安全门控与模型选择

W0 inner-validation 基线通过 patient-level bootstrap 建立一个标准误差的非劣边界，且每项 margin 不小于 0.005。候选 checkpoint 必须同时满足：ACS Macro AUPRC；OMI AUPRC、PPV、Specificity；STEMI F1；ACS/OMI Brier；以及相对 W0 的 mean KL 上限。

通过 safety gate 后，再按预注册 inner-validation utility（NSTEMI F1 + OMI F1 + NSTE-ACS 中 OMI AUPRC）选择 checkpoint。outer-test 不参与任何选择。

## 4. 实现与审计证据

- 策略：`experiments/ecg_lfm/rl/clinical_policy.py`
- Reward：`experiments/ecg_lfm/rl/clinical_reward.py`
- KL：`experiments/ecg_lfm/rl/kl_regularization.py`
- 安全约束：`experiments/ecg_lfm/rl/clinical_constraints.py`
- Patient bootstrap：`experiments/ecg_lfm/rl/reference_guard.py`
- Fold0 训练：`experiments/ecg_lfm/train_rl_fold.py`
- Outer 一次性评估：`experiments/ecg_lfm/evaluate_rl_outer_once.py`
- 配置：`configs/ecg_lfm/rl/`
- 单元测试：`tests/test_ecg_lfm_clinical_rl.py`

服务器真实 ECG-LFM 环境单元测试：15/15 通过。覆盖任务因子化、8 动作空间、独立 OMI Sigmoid、初始化 KL=0、KL 非负、奖励条件、Soft-F1 梯度、LoRA+Head 冻结边界、patient-level bootstrap 以及禁止训练脚本读取 outer-test。

## 5. Fold0 Smoke Test

真实 W0 的 2-update GPU smoke 已完成：15/15 单元测试先通过，随后真实 ECG-LFM forward/backward 无 NaN/Inf。正式网格单组耗时约 301–332 秒，峰值显存均为 16.41 GiB。每个候选均从同一个 W0 重新初始化，Reference 与 Policy 初始化概率一致且 KL=0；训练脚本没有加载 outer-test。

## 6. Inner-validation 探索结果

预注册小网格 8/8 完成，8/8 通过 patient-bootstrap safety gate。结果按“safety 通过后 clinical utility 降序”列出：

| LR | KL | Hard scale | ACS Macro AUPRC | NSTEMI AUPRC | NSTEMI F1 | OMI AUPRC | OMI F1 | OMI Sens. | OMI Spec. | OMI PPV | NSTE-OMI AUPRC | mean KL | Utility |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-6 | 0.01 | 0.5 | 0.55791 | 0.31272 | 0.33125 | 0.46500 | 0.43490 | 0.86528 | 0.84708 | 0.29043 | 0.31126 | 0.00698 | 1.07741 |
| 5e-6 | 0.05 | 0.5 | 0.55784 | 0.31263 | 0.32704 | 0.46511 | 0.43546 | 0.86528 | 0.84745 | 0.29094 | 0.31135 | 0.00668 | 1.07386 |
| 5e-6 | 0.05 | 1.0 | 0.55799 | 0.31305 | 0.33229 | 0.46471 | 0.42184 | 0.88083 | 0.83396 | 0.27732 | 0.31054 | 0.00577 | 1.06466 |
| 5e-6 | 0.01 | 1.0 | 0.55798 | 0.31332 | 0.33229 | 0.46509 | 0.42079 | 0.88083 | 0.83321 | 0.27642 | 0.31083 | 0.00592 | 1.06391 |
| 1e-6 | 0.01 | 1.0 | 0.55854 | 0.31435 | 0.32131 | 0.46572 | 0.42948 | 0.87565 | 0.84070 | 0.28451 | 0.31139 | 0.00032 | 1.06218 |
| 1e-6 | 0.05 | 1.0 | 0.55859 | 0.31454 | 0.32131 | 0.46657 | 0.42803 | 0.87047 | 0.84108 | 0.28378 | 0.31156 | 0.00032 | 1.06090 |
| 1e-6 | 0.05 | 0.5 | 0.55843 | 0.31396 | 0.32131 | 0.46547 | 0.42694 | 0.87047 | 0.84033 | 0.28283 | 0.31104 | 0.00034 | 1.05929 |
| 1e-6 | 0.01 | 0.5 | 0.55845 | 0.31377 | 0.31579 | 0.46622 | 0.42948 | 0.87565 | 0.84070 | 0.28451 | 0.31117 | 0.00035 | 1.05644 |

W0 在同一 runner 重新计算的参考值：ACS Macro AUPRC 0.55838、NSTEMI AUPRC 0.31389、NSTEMI F1 0.32450、OMI AUPRC 0.46641、OMI F1 0.40952、OMI Sensitivity 0.89119、Specificity 0.82196、PPV 0.26584、NSTE-OMI AUPRC 0.31083、Utility 1.04486。

按预注册选择规则，选中 `LR=5e-6, KL=0.01, hard scale=0.5` 进入完整 Fold0。需要谨慎解释：该候选主要提高了阈值相关的 OMI F1/PPV/Specificity 和 NSTEMI F1，但 ACS/OMI primary AUPRC 均较 W0 略低，只能称为 safety 范围内的短步候选，尚不能称为 RL 涨点。

## 7. Fold0 冻结 Outer-test 结果

只有 inner-validation 规则选出并冻结 checkpoint 后，本节才允许写入一次 outer-test 结果与 calibration 结果。

## 8. 消融实验

- R1：Clinical RL + KL；
- R2：去掉 KL；
- R3：去掉 NSTEMI/NSTE-OMI hard-case reward。

消融是否启动取决于 R1 Fold0 safety gate。若 R1 明确 NO-GO，不启动完整五折。

## 9. 最终判定

实施中，尚不能声明 RL 有效。Fold0 若不能在保持 ACS/OMI primary performance、PPV/Specificity 与 calibration 的同时改善目标困难病例，则结论为 RL NO-GO，并停止 full 5-fold。
