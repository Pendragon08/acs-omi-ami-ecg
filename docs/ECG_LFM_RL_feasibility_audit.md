# ECG-LFM 强化学习后训练可行性审计

> 审计日期：2026-08-31
> 审计范围：当前工作区中的 ECG-LFM 代码、ACS/OMI 联合任务代码、配置、患者级折分、训练日志、OOF 汇总、外部平台汇总、训练成本记录及现有报告。
> 本轮行为：只读扫描与证据整理；未修改训练代码、Loss 或 fold，未运行 GPU 训练，未下载或重新预处理数据，也未实现任何 RL trainer。
> 最终结论：**NO-GO——当前不应进入 RL post-training。**

审计证据总评为 **WARN / provisional**：没有发现伪造真值、用预测自身最大值归一化分数或本地读取外部隐藏标签，但验证协议和 provenance 尚不足以支撑确认性结论。

## 证据口径与限制

本报告优先使用当前 Git 仓库 `acs-omi-ami-ecg` 中的 source/config/output，并用工作区根目录的预处理完整性文件补充数据来源证据。旧四标签独立 sigmoid 实验只用于辨认历史遗留，绝不与当前联合任务结果混合。

主要证据路径：

- 任务与数据：`DATASET.md`、`docs/plans/CORRECTED_TASK_FORMULATION.md`；
- 当前配置：`configs/ecg_lfm/corrected_joint_{baseline,lora,dora}.yaml`；
- 模型与 Loss：`third_party/ECG-LFM/fairseq_signals/models/classification/ecg_transformer_classifier.py`、`third_party/ECG-LFM/fairseq_signals/criterions/acs_joint.py`；
- 训练与评估：`experiments/ecg_lfm/run_joint_fold.py`、`experiments/ecg_lfm/evaluate_joint.py`、`experiments/acs_joint_task.py`、`scripts/run_corrected_ecg_lfm_joint.sh`；
- 当前结果：`results/corrected_ecg_lfm_joint_bs64_v2/{baseline,lora,dora}`；
- 外部平台：各方法目录下的 `official_platform_metrics.json`；
- 预处理完整性：工作区根目录 `artifacts/ecg_lfm/raw_500hz_download_report.json`、`preprocess_integrity.json`、`cohort_integrity.json`。

存在三项必须提前说明的限制：

1. 当前三个 corrected ECG-LFM 汇总都包含五折 checkpoint 哈希和 pooled OOF 指标，但没有保存逐折临床预测指标或可重新汇总的 OOF 逐例概率。因此无法从本地诚实补算临床指标的 `fold mean ± std`；只有 pooled OOF 和训练日志中的逐折 validation loss/accuracy 可用。
2. 现存 `summary.json` 记录的 evaluator SHA256 为 `c2b80b...95cac`，而当前 `experiments/ecg_lfm/evaluate_joint.py` 为 `eda0c2...24859`；当前 `experiments/acs_joint_task.py` 为 `d4524c...3a06`，旧汇总没有记录该哈希。因此现存输出并非由当前评估器版本生成。
3. 旧评估器曾用 hard argmax 指示量计算“限定 ACS 阳性三分类”的 AUROC/AUPRC，并在不存在可行 `Sensitivity >= 0.95` 阈值时仍返回一个数值。故本报告不把旧 conditional ACS AUROC/AUPRC 和 LoRA/DoRA 的 `sens95` 当作有效结果；conditional hard confusion/F1/accuracy 仍可使用。四状态 ACS 的 AUROC/AUPRC 原本即由连续 softmax 概率计算，未受该项旧缺陷影响。
4. 当前 ECG-LFM OOF 的真实标签由五列 `y_all.npy` 按 master `idx` 位置读取，并有 fold、patient overlap、coverage 和文件哈希检查；但没有带 record ID 的标签表逐 ID join。这是身份对齐的剩余 provenance 风险，不代表已发现错位。外部平台仓库内只保留聚合 JSON 和 submission hash，没有原始平台响应或 submission CSV，故只能审计聚合结果。

# 1. Current pipeline

当前端到端流程为：已物化的 500 Hz、12 导联 ECG → 每条记录两个 5 秒 segment → ECG-LFM encoder → 时间维平均池化 → 一个 `Linear(1024, 5)` 投影 → 前四维作为 `OTHER/STEMI/NSTEMI/UA` categorical logits，第五维作为 OMI logit → segment logits 按记录平均 → ACS softmax/argmax 与 OMI sigmoid/阈值。

本地事实与背景描述的差异如下：

- 语义上确实是 ACS head + OMI head，但代码不是两个独立 `nn.Module` head，而是一个五输出线性层再按 `[:4]` 和 `[4]` 切分。证据：`ecg_transformer_classifier.py:41`、`acs_joint.py:73-80`。
- Baseline 不是严格“所有参数全参微调”：配置 `feature_grad_mult: 0.0`，前端特征提取部分不回传梯度；其余主干大部分参数与联合分类头参与训练。因此更准确叫“部分/有效微调”。证据：三个 corrected config 的 `model.feature_grad_mult` 以及 ECG-LFM feature extractor 的 `no_grad` 路径。
- config 中 `normalize: false`、`filter: false` 不等于原始波形从未处理，而是训练输入已经预先物化。本地 `preprocess_integrity.json` 记录了 1–30 Hz 二阶 Butterworth、50/60 Hz notch、按记录/导联 z-score、拆成两个 5 秒 segment；共 19,955 条、39,910 segments、0 error。
- LoRA/DoRA 向 encoder layer 下的线性层注入 rank-8 adapter，冻结 base weight，同时保持联合分类层可训练；实现见 `parameter_efficient.py:120-190` 和 `ecg_transformer_classifier.py:46-57`。
- `ecg_transformer_classifier.py:59-64` 还保留一个 `get_logits(normalize=True)` 分支，会错误地把五个 logits 一起 log-softmax，使 OMI 与 ACS 状态耦合。当前 criterion/evaluator 使用默认 `normalize=False`，所以该死分支未污染现存结果，但应在未来代码维护中单独处理；本轮未按约束修改。

当前训练配置交叉核对如下：

| 项目 | Baseline | LoRA | DoRA | 证据 |
|---|---:|---:|---:|---|
| 模型输出 | 5 logits | 5 logits | 5 logits | config `num_labels: 5`；`ecg_transformer_classifier.py:41` |
| 微调策略 | adapter=none，前端 feature grad=0 | rank 8, alpha 16, dropout 0.05 | rank 8, alpha 16, dropout 0.05 | corrected configs `:71,74-77` |
| Loss | CE + BCEWithLogits | 同左 | 同左 | `acs_joint.py:73-76` |
| ACS class weight | null | null | null | corrected configs `:43` |
| OMI pos_weight | null | null | null | corrected configs `:44` |
| task loss weight | 1.0 | 1.0 | 1.0 | corrected configs `:45` |
| learning rate | 1e-6 | 1e-4 | 5e-5 | corrected configs `:51` |
| batch size | 64 | 64 | 64 | corrected configs `:32` |
| gradient accumulation | 4 | 4 | 4 | corrected configs `:50`；有效 batch 256 |
| optimizer | Adam, betas=(0.9,0.98), eps=1e-8 | 同左 | 同左 | corrected configs `:52-55` |
| scheduler | step decay，period 112，factor 0.99，min_lr 1e-12，warmup 0 | 同左 | 同左 | corrected configs `:56-61` |
| max epochs | 100 | 100 | 100 | corrected configs `:48` |
| early stopping | patience 10 | 10 | 10 | corrected configs `:11` |
| checkpoint criterion | 最小 validation total loss | 同左 | 同左 | corrected configs `:9-10` |
| seed | 2026 | 2026 | 2026 | runner 与数据说明 |

注意：三种方法同时改变了微调方式和 learning rate，因此这是工程方案比较，不是“只改变 adapter”的严格因果消融。

# 2. Current task definition

当前正确任务是两个并行但不同类型的监督任务：

1. 状态分类：全训练队列保留 `[0,0,0]` 为 `OTHER/non-ACS`，使用 `OTHER/STEMI/NSTEMI/UA` 四状态 softmax；其中 STEMI/NSTEMI/UA 互斥，OTHER 不是第四种 ACS 亚型。
2. 病理实质：独立的 OMI/NOMI 二分类，允许与状态标签组合。

训练目标为：

`L = CrossEntropy(ACS_state) + 1.0 * BCEWithLogits(OMI)`。

ACS 预测使用 argmax，不存在每类独立 Youden 阈值；只有 OMI 使用阈值。代码证据见 `acs_joint.py:73-80`，方案证据见 `docs/plans/CORRECTED_TASK_FORMULATION.md:18-22`。

数据事实：

| 项目 | 数量 |
|---|---:|
| 训练 ECG | 17,960 |
| 训练患者 | 17,018 |
| OTHER | 9,070 |
| STEMI | 1,442 |
| NSTEMI | 1,235 |
| UA | 6,213 |
| OMI 阳性 | 1,151 |
| 隔离/外部平台 ECG | 1,995 |

证据：`DATASET.md:19-28`。患者级五折使用 seed=2026，五折验证记录数分别为 3,606、3,582、3,596、3,603、3,573；本地 fold integrity 记录 patient overlap 为 0。

# 3. Existing experiments

## 3.1 当前正确联合任务

| 实验 | 当前任务正确 | 五折状态 | 可用输出 | 判定 |
|---|---|---|---|---|
| ECG-LFM Baseline/PFT | 是 | Fold0–4 完成 | 五折日志、五个 checkpoint 哈希、pooled OOF、曲线、CM、外部平台汇总 | 完整五折 |
| ECG-LFM LoRA | 是 | Fold0–4 完成 | 同上 | 完整五折 |
| ECG-LFM DoRA | 是 | Fold0–4 完成 | 同上 | 完整五折 |
| E4 HMoE/KAN corrected | 是 | Fold0–4 完成 | `results/corrected_joint_acs_omi_e4` | 完整五折 |
| InceptionTime | 是 | Fold0–4 完成 | `results/corrected_joint_model_comparison/InceptionTime` | 完整五折 |
| ResNet1D-Wang | 是 | Fold0–4 完成 | 对应模型目录 | 完整五折 |
| LSTM | 是 | Fold0–4 完成 | 对应模型目录 | 完整五折 |
| ViT | 是 | Fold0–4 完成 | 对应模型目录 | 完整五折 |
| Transformer1D | 是 | Fold0–4 完成 | 对应模型目录 | 完整五折 |

当前 corrected ECG-LFM 三个方案没有只完成 Fold0 的方法。它们都只有一个随机种子 2026，不能把五个患者折当作五个独立随机种子。

## 3.2 不能混入当前结论的历史结果

- 工作区根目录 `results/ecg_lfm_4label/finetune_paper`：虽有 Fold0–4，但任务是 STEMI/NSTEMI/UA/OMI 四个独立 sigmoid，已被当前任务定义否定。
- `artifacts/ecg_lfm/tuning_fold0_comparison_20260828.*`、`linear_probe_fold0_reprioritization.json` 等：旧错误任务的 Fold0 screening/诊断，不是当前五折结果。
- `results/loss_feature_ablation`：五 seed 的旧 OMI/AMI endpoint，与当前 ACS-state+OMI ECG-LFM 联合任务不同，不能证明当前 CB-Focal、weighted loss 等已经验证。
- 工作区根目录的旧 configs/reports 是清理前镜像；当前 Git 仓库的 corrected source/output 优先级更高。

# 4. Current best model

**当前工程推荐最优是 ECG-LFM LoRA**：rank=8、alpha=16、dropout=0.05、LR=1e-4、batch=64、gradient accumulation=4。

“最优”不是按一个预注册 primary endpoint 得出，而是多指标工程选择：LoRA 在 pooled OOF 的四状态 ACS macro AUROC、macro AUPRC、macro F1、balanced accuracy、条件 ACS 三分类 macro F1，以及 OMI AUROC/AUPRC 上领先；外部平台四个 one-vs-rest F1 的简单平均也最高，并且报告的训练成本低于 DoRA。DoRA 在 OMI Youden 工作点 F1/PPV/specificity、条件 ACS accuracy 和外部 OMI/UA F1 略高。

因此当前只能说“LoRA 是整体折中最优”，不能说它在所有指标上显著优于 DoRA，也没有多 seed 或 paired CI 支持统计显著性。

# 5. Five-fold / held-out performance

## 5.1 pooled OOF 主结果

下表的 ACS 是全 17,960 队列上的 `OTHER/STEMI/NSTEMI/UA` 四状态；OMI hard metrics 采用 pooled OOF Youden 阈值。阈值和 hard metrics 在同一 pooled OOF 上选择/计算，属于描述性 OOF，不是严格 nested-CV 无偏估计。

| Method | ACS macro AUROC | ACS macro AUPRC | ACS macro F1 | ACS balanced acc | ACS acc | OMI AUROC | OMI AUPRC | OMI F1 | Sens | Spec | PPV | NPV | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline/PFT | 0.7826 | 0.5329 | 0.4994 | 0.4954 | 0.5684 | 0.9052 | 0.4159 | 0.3826 | 0.8488 | 0.8228 | 0.2470 | 0.9876 | 0.05 |
| **LoRA** | **0.7870** | **0.5390** | **0.5103** | **0.5053** | **0.5695** | **0.9106** | **0.4315** | 0.4021 | **0.8549** | 0.8359 | 0.2629 | **0.9883** | 0.05 |
| DoRA | 0.7838 | 0.5369 | 0.5069 | 0.5024 | 0.5680 | 0.9094 | 0.4267 | **0.4054** | 0.8532 | **0.8387** | **0.2658** | 0.9882 | 0.05 |

LoRA 相对 Baseline：ACS macro AUROC +0.0045、macro AUPRC +0.0060、macro F1 +0.0108；OMI AUROC +0.0054、AUPRC +0.0157、Youden F1 +0.0195、Sensitivity +0.0061。提升幅度较小且仅一个 seed，尚不能称为稳定或统计显著。

## 5.2 四状态 ACS 逐类 pooled OOF

| Method | Class | AUROC | AUPRC | F1 |
|---|---|---:|---:|---:|
| Baseline | OTHER | 0.6976 | 0.6873 | 0.6638 |
| Baseline | STEMI | 0.9467 | 0.6911 | 0.6440 |
| Baseline | NSTEMI | 0.8141 | 0.2717 | 0.2604 |
| Baseline | UA | 0.6718 | 0.4816 | 0.4295 |
| **LoRA** | OTHER | 0.7011 | 0.6908 | 0.6573 |
| **LoRA** | STEMI | 0.9491 | 0.7017 | 0.6477 |
| **LoRA** | NSTEMI | 0.8230 | 0.2805 | 0.2813 |
| **LoRA** | UA | 0.6750 | 0.4830 | 0.4548 |
| DoRA | OTHER | 0.6982 | 0.6878 | 0.6579 |
| DoRA | STEMI | 0.9497 | 0.6972 | 0.6542 |
| DoRA | NSTEMI | 0.8155 | 0.2825 | 0.2748 |
| DoRA | UA | 0.6716 | 0.4799 | 0.4408 |

四状态 confusion matrix，行是真值、列是预测，顺序均为 `[OTHER, STEMI, NSTEMI, UA]`：

- Baseline：`[[6605,238,136,2091],[263,967,105,107],[446,207,244,338],[3517,149,154,2393]]`
- LoRA：`[[6347,230,184,2309],[230,954,151,107],[421,196,291,327],[3245,124,208,2636]]`
- DoRA：`[[6441,249,156,2224],[224,980,118,120],[441,192,266,336],[3405,133,161,2514]]`

## 5.3 限定 ACS 阳性的三亚型 hard 结果

旧汇总中的 conditional AUROC/AUPRC 不可靠，故只列不依赖旧 hard-score AUC 实现的结果。

| Method | ACS3 accuracy | Macro F1 | Balanced acc | STEMI F1 | NSTEMI F1 | UA F1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.8055 | 0.6439 | 0.6316 | 0.7083 | 0.3275 | 0.8958 |
| **LoRA** | 0.8058 | **0.6541** | **0.6400** | 0.7136 | **0.3510** | 0.8977 |
| DoRA | **0.8092** | 0.6522 | 0.6377 | **0.7200** | 0.3387 | **0.8980** |

LoRA conditional ACS confusion matrix `[STEMI,NSTEMI,UA]` 为 `[[1004,188,250],[216,358,661],[152,259,5802]]`。1,235 个 NSTEMI 中只有 358 个判对，661 个被判为 UA，是当前最明显的亚型混淆。

## 5.4 OMI threshold trade-off

| Method/工作点 | Threshold | Sens | Spec | PPV | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline fixed 0.5 | 0.50 | 0.2354 | 0.9860 | 0.5345 | 0.3269 |
| Baseline max F1 | 0.18 | 0.6160 | 0.9299 | 0.3757 | 0.4668 |
| Baseline Youden | 0.05 | 0.8488 | 0.8228 | 0.2470 | 0.3826 |
| LoRA fixed 0.5 | 0.50 | 0.2824 | 0.9830 | 0.5328 | 0.3691 |
| **LoRA max F1** | **0.25** | 0.5543 | 0.9460 | 0.4127 | 0.4731 |
| **LoRA Youden** | **0.05** | 0.8549 | 0.8359 | 0.2629 | 0.4021 |
| LoRA Sens≥0.90 | 0.02 | 0.9192 | 0.7420 | 0.1961 | 0.3233 |
| DoRA max F1 | 0.18 | 0.6386 | 0.9296 | 0.3832 | **0.4790** |
| DoRA Youden | 0.05 | 0.8532 | 0.8387 | 0.2658 | 0.4054 |
| DoRA Sens≥0.90 | 0.02 | 0.9105 | 0.7472 | 0.1978 | 0.3251 |

LoRA Youden OMI confusion 为 TN=14,050、FP=2,759、FN=167、TP=984。该表显示主要问题不是完全无法排序，而是临床工作点选择：同一模型从 threshold 0.5 调到 0.05，Sensitivity 可由 0.282 升至 0.855，但 PPV 由 0.533 降至 0.263。

## 5.5 fold mean ± std 的实际可用范围

当前 summary 的 `folds` 只保存记录数和 checkpoint/hash，不保存逐折 AUROC/AUPRC/F1。因此以下是从 sanitized validation logs 得到的训练过程统计，不应替代临床指标的 fold mean ± std：

| Method | best val total loss, mean ± std | best epoch, mean ± std | best-loss epoch ACS acc | best-loss epoch OMI acc |
|---|---:|---:|---:|---:|
| Baseline | 1.0493 ± 0.0101 | 5.4 ± 1.14 | 0.5656 ± 0.0051 | 0.9370 ± 0.0015 |
| LoRA | **1.0414 ± 0.0083** | 4.2 ± 0.84 | **0.5663 ± 0.0055** | 0.9369 ± 0.0021 |
| DoRA | 1.0435 ± 0.0103 | 7.2 ± 2.68 | 0.5632 ± 0.0120 | **0.9384 ± 0.0009** |

accuracy 在不平衡任务中受多数类主导，不能作为主临床结论。要补齐用户要求的 fold mean ± std，未来评估阶段必须保存每折 record-level probability 或每折完整 metric JSON；本轮按约束没有重新推理或训练。

## 5.6 held-out/external platform

外部平台只有每列 one-vs-rest hard-label 指标，不返回概率、联合四状态 confusion matrix 或 AUROC/AUPRC；不能据此构造严格的外部 ACS 三分类/四状态总体 AUROC、AUPRC、macro F1 或 accuracy。外部平台已被多方案反复查看并参与选型，故更准确叫 held-out platform evaluation，不再是完全未触碰的 confirmatory test。

| Method | Label | Sens | Spec | PPV | NPV | Accuracy | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | STEMI | 0.7222 | 0.9684 | 0.6686 | 0.9753 | 0.9484 | 0.6944 |
| Baseline | NSTEMI | 0.1397 | 0.9736 | 0.2794 | 0.9393 | 0.9168 | 0.1863 |
| Baseline | UA | 0.3783 | 0.7814 | 0.4734 | 0.7076 | 0.6436 | 0.4205 |
| Baseline | OMI | 0.8672 | 0.8211 | 0.2494 | 0.9890 | 0.8241 | 0.3874 |
| **LoRA** | STEMI | 0.7469 | 0.9744 | 0.7202 | 0.9776 | 0.9559 | **0.7333** |
| **LoRA** | NSTEMI | 0.2132 | 0.9699 | 0.3412 | 0.9440 | 0.9183 | **0.2624** |
| **LoRA** | UA | 0.4047 | 0.7730 | 0.4808 | 0.7143 | 0.6471 | 0.4395 |
| **LoRA** | OMI | 0.8672 | 0.8270 | 0.2558 | 0.9891 | 0.8296 | 0.3950 |
| DoRA | STEMI | 0.7346 | 0.9744 | 0.7169 | 0.9765 | 0.9549 | 0.7256 |
| DoRA | NSTEMI | 0.1912 | 0.9736 | 0.3467 | 0.9427 | 0.9203 | 0.2464 |
| DoRA | UA | 0.4047 | 0.7814 | 0.4902 | 0.7165 | 0.6526 | **0.4434** |
| DoRA | OMI | 0.8594 | 0.8329 | 0.2607 | 0.9886 | 0.8346 | **0.4000** |

# 6. Error and bottleneck analysis

| 类型 | 判断 | 本地证据 | 解释 |
|---|---|---|---|
| A. Representation | 部分存在，但不是 OMI 主瓶颈 | LoRA OMI AUROC 0.9106、STEMI AUROC 0.9491；UA AUROC 0.6750，conditional NSTEMI F1 0.3510 | 表征足以完成 OMI 排序和 STEMI 辨别，但对 UA/NSTEMI 细粒度边界仍弱 |
| B. Class imbalance | **明确存在** | NSTEMI 1,235/17,960，OMI 1,151/17,960；当前 CE/BCE 均未加权；NSTEMI AUPRC 0.2805、四状态 sensitivity 0.2356 | 少数类梯度和 hard decision 明显弱于多数类；这是应先验证监督加权的直接理由 |
| C. Threshold/calibration | **OMI 的主要瓶颈之一** | OMI AUROC 0.9106，但 threshold 0.5/0.25/0.05 的 Sens、PPV、F1 大幅变化；无 Brier/ECE | 排序能力较高，工作点却决定了临床性能；当前没有校准证据 |
| D. Multi-task interference | 未证实 | 没有同 split 的 ACS-only、OMI-only 对照 | 不能声称一个任务拖累另一个任务 |
| E. Fine-tuning strategy | 有影响但幅度有限 | LoRA 对 PFT 的 ACS macro F1 +0.0108、OMI AUPRC +0.0157；DoRA 优劣混合；三者 LR 不同 | PEFT 有工程价值，但现有比较混入 LR 影响，不能作纯 adapter 因果结论 |
| F. Label/task information ceiling | 可能存在，未由本地实验确认 | conditional NSTEMI 中 661/1,235 被判 UA；项目没有 troponin/临床变量对照 | ECG-only 对 NSTEMI/UA 的信息上限是合理临床假设，但当前不能把它写成已证实事实 |
| G. Optimization | **存在快速过拟合** | LoRA 五折 best epoch 为 3–5，patience 后最后 val loss 约 1.51–1.73，而 best 约 1.03–1.05 | early stopping 保护了 checkpoint，但 LR/正则/adapter 配置仍有优化空间；不是非得用 RL 的问题 |

三种方法的 best/last validation loss 进一步支持过拟合判断：Baseline 最后约 1.19–1.27，LoRA 最后约 1.51–1.73，DoRA 最后约 1.09–1.18；LoRA 恶化最明显。因为 LoRA 同时使用最高 LR=1e-4，不能把原因完全归于 LoRA 结构。

当前没有有效的 `ACS vs OTHER` 单独阈值结果、`NSTEMI vs UA` 专门二分类结果、`OMI among NSTE-ACS` slice、calibration/Brier/ECE，也没有 joint vs single-task ablation。这些缺失限制了对瓶颈的进一步归因。

# 7. Fine-tuning comparison

## 7.1 训练成本

现有总报告给出的成本如下，但必须标记为“报告级历史估计”，而不是当前五输出 checkpoint 的重新审计值：原参数统计来自旧四输出 artifact；当前 linear head 比旧 head 多 1,025 个参数，现有 corrected summary 又没有保存参数审计、原始显存或 wall-clock provenance。

| Method | 报告总参数 | 报告可训练参数 | 可训练比例 | 报告峰值显存 | 报告时间/fold | 证据强度 |
|---|---:|---:|---:|---:|---:|---|
| Baseline/PFT | 622,880,004 | 622,479,108 | 99.94% | 约 35.4 GiB | 6,670.3 ± 495.7 s | 报告值，head 计数旧一版 |
| LoRA | 628,778,244 | 5,902,340 | 0.9389% | 约 28.4 GiB | 6,131.3 ± 367.1 s | 同左 |
| DoRA | 629,146,884 | 6,270,980 | 0.9967% | 约 31.3 GiB | 7,886.6 ± 1,674.8 s | 同左 |

若 backbone 不变，仅按 `Linear(1024,4) → Linear(1024,5)` 结构差异机械修正，理论上三者 total/trainable 都应各增加 1,025；但本报告不把这个推导数冒充实测值。后续如要发表训练成本结论，应在 current corrected checkpoint 环境重新导出 resolved config、参数计数、`torch.cuda.max_memory_allocated` 和 wall-clock 元数据。

## 7.2 性能—成本判断

LoRA 用约 0.94% 报告可训练参数取得整体最佳 pooled OOF 折中，PEFT 的训练成本优势是现有证据支持的工程贡献。但性能增益小、只有一个 seed、LR 不公平，不能把“LoRA 本身显著涨点”写成确定科学结论。DoRA 没有显示一致优势，其训练时间反而更长。

# 8. Supervised methods not yet exhausted

当前 corrected joint config 明确是 unweighted CE + unweighted BCE、任务权重 1:1。虽然 criterion/runner 具备部分权重入口，但主五折 runner 没有实际使用。旧 OMI/AMI loss ablation 不是当前任务证据。

| 监督方法 | 当前联合 ECG-LFM 是否完成公平验证 | 本地证据/缺口 |
|---|---|---|
| class-weighted CrossEntropy | 否 | `acs_weight: null` |
| OMI `pos_weight` BCE | 否 | `omi_pos_weight: null`；runner 有入口但主脚本未传 |
| Focal Loss | 否 | 只有旧 endpoint 配置/结果 |
| Asymmetric Loss | 否 | 只有旧 endpoint 实现/结果 |
| label smoothing | 否 | 当前 criterion 未启用 |
| hard-negative mining | 否 | 无 current joint 实验 |
| task loss reweighting | 否 | 只用 `omi_loss_weight=1.0` |
| uncertainty weighting | 否 | 无实验 |
| calibration | 否 | 无 temperature/Platt/isotonic/beta calibration；无 Brier/ECE |
| nested per-task threshold | 否 | pooled OOF 同一批标签选阈值并评分 |
| cost-sensitive loss/cost matrix | 否 | 无实验 |
| differentiable F1/utility | 否 | 无实验 |
| ACS-only / OMI-only vs joint | 否 | 无法判断 multi-task interference |

因此监督优化远未穷尽。现阶段直接进入 RL 会跳过更简单、方差更低、可解释性更高且与已确认问题直接匹配的 baselines。

# 9. Is RL scientifically justified?

**当前不具备科学必要性。**

1. 当前 `ACS CrossEntropy + OMI BCEWithLogits` 已经能表达“预测正确类别/标签”这一基本目标，且概率排序并不差：OMI AUROC 0.91，STEMI AUROC约 0.95。
2. 本地确实出现可能具有不同临床代价的错误：OMI FN 可能比 FP 更严重，NSTEMI→UA 是高频混淆。但项目没有预注册临床 cost matrix、PPV 下限或联合 utility，也没有证明 weighted/cost-sensitive supervised loss 和校准无法处理这些代价。
3. 当前最强证据指向 class imbalance、operating threshold/calibration、fast overfitting 和可能的信息上限。这些都不是 RL 特有问题。
4. 外部平台已参与模型选择，当前又只有单 seed；在证据基础尚不稳时加入 RL 会更难判断提升来源。

所以，不能因为 RL 更新颖就推荐 RL，也不能把“O​​MI Sensitivity 想更高”自动等价为 RL 问题。

# 10. If RL is justified what exactly should it optimize?

当前结论是 NO-GO，因此以下只定义未来重新评估 RL 时必须满足的具体目标，不是实施方案。

只有当临床团队先明确一个不可简单分解的联合 utility，并且监督 Pareto frontier 已经失败，RL 才可能针对：

- 在预先给定 PPV/Specificity 下限的同时降低 OMI false negative；
- 在不降低 ACS macro F1/尤其 STEMI performance 的前提下改善 OMI sensitivity–PPV trade-off；
- 对 NSTE-ACS 子集内的 OMI detection 赋予特殊代价；
- 对 `NSTEMI→UA`、`STEMI→OTHER`、OMI FN 等不同联合错误定义非对称、非可分的临床 utility。

不能把目标写成泛泛的“提高总性能”。当前尚未有 NSTE-ACS slice 指标或临床代价矩阵，所以这些只是假设性未来目标，不是已确认的 RL 必要性。

# 11. Why not simpler supervised losses?

当前没有证据说明 CE/BCE 的可加权版本不够。若错误代价是固定的，例如 OMI FN 权重大于 FP，可以直接使用 train-fold-only `pos_weight`、成本敏感 BCE、阈值约束或校准；NSTEMI 少数类可先用 class-weighted CE、Focal/Asymmetric loss 或专门层级分类对照；任务竞争可用固定/uncertainty task weighting。

这些监督方法直接优化 logits，梯度方差小、训练稳定、概率校准更容易审计，而且能做严格的单因素比较。只有当目标是多个相互竞争、不可微且不能合理写成样本级损失的临床 utility，并且这些监督方法在多 seed/nested validation 中确实无法到达目标 Pareto front，RL 才新增实质能力。

# 12. Risks of RL

当前输出可以形式化为：

`π(a|x) = Categorical(ACS_state | x) × Bernoulli(OMI | x)`。

数学上可以定义 policy，但每条 ECG 只是一次性决策，联合 action space 仅 `4 × 2 = 8`，没有时序状态转移。这更接近 contextual bandit/成本敏感分类，而不是需要 PPO/GRPO/DAPO 的序列 RL。action variability 有限，奖励大概率可以直接转成 supervised loss。

主要风险：

- policy-gradient 高方差，单 seed 下更容易产生偶然涨点；
- reward hacking：为了高 sensitivity 大量预测 OMI，导致 PPV/Specificity 崩溃；
- 概率 calibration 进一步变差；
- catastrophic forgetting，破坏 ECG-LFM 已学到的表征；
- OMI 奖励压过 ACS，牺牲 STEMI/NSTEMI/UA；
- 小 action space 下收益不超过成本敏感监督学习，却显著增加训练成本和调参自由度；
- 若复用已参与选型的外部平台，会进一步放大 test overfitting。

# 13. Required experiments before RL

建议按以下顺序完成监督证据链；这不是本轮要启动的训练任务：

1. **先修正评估设计**：在每个 outer-train 内建立 patient-level inner validation，用于 checkpoint、超参数和 OMI threshold/calibration；outer fold 只评估一次。否则 pooled OOF hard metrics 继续是描述性结果。
2. **当前 joint LoRA 的严格单因素 loss baseline**：固定 split/seed/preprocessing/model/init/optimizer/scheduler/batch，只比较 unweighted、ACS class-weighted CE、OMI fold-specific `pos_weight` BCE、两者组合、Focal/Asymmetric loss。
3. **task weighting**：验证 OMI loss lambda 和 uncertainty weighting，记录 ACS 与 OMI 的 Pareto front。
4. **校准与 operating point**：ACS temperature scaling；OMI Platt/isotonic/beta calibration；报告 Brier、ECE、calibration slope/intercept；由临床先给定 Sensitivity/PPV/Specificity 目标。
5. **归因消融**：同 split 的 ACS-only、OMI-only、joint；专门报告 NSTEMI vs UA、OMI among NSTE-ACS、ACS vs OTHER。
6. **优化稳定性**：LoRA LR/regularization 小网格，重点解决 epoch 3–5 后 val loss 快速恶化；至少 3–5 个独立 seed，并给 paired CI。
7. **确认性验证**：冻结方案后使用一个未参与选型的外部队列；现有 1,995 条平台结果只作为已使用 held-out evidence。

未来重新考虑 RL 的最低门槛：上述 supervised baselines 和 calibration 已完成；多 seed 结果稳定；临床 utility 预注册且不能化为简单样本级 cost；监督方法无法达到由临床指定的 sensitivity–PPV/ACS performance Pareto 目标；另有 untouched confirmatory set。即便满足，也应先比较 differentiable constrained optimization 或 contextual bandit，再考虑 PPO/GRPO/DAPO。

# 14. Final GO / CONDITIONAL GO / NO-GO recommendation

## **NO-GO**

当前问题用监督学习、阈值/校准、cost-sensitive loss、task weighting 和更严格的验证设计仍有充分解决空间；本地没有任何实验表明这些方法已经失败，也没有预注册的非可分 clinical utility。现有分类动作空间小且非序列，RL 很可能只是用更高方差、更高成本的方法近似一个本可直接写成监督损失的目标。

当前最合理的下一实验不是 RL，而是以 LoRA 为工程基线，先进行 **patient-level nested calibration + class-weighted CE / fold-specific OMI pos_weight BCE 的严格单因素对比**，同时补齐 Brier/ECE、NSTEMI-vs-UA、OMI-among-NSTE-ACS、single-task vs joint 和多 seed 统计。

若这些监督方法在预注册临床约束下形成明确但无法突破的 Pareto frontier，再重新审议 RL；在那之前，不建议为了创新点强行加入 RL。
