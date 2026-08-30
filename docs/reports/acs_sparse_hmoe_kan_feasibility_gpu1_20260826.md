---
experiment_id: ACS-SPARSE-HMOE-KAN-FULLCOHORT-GPU1-20260826
date: 2026-08-26
status: completed (LEGACY—three-independent-sigmoid task)
task: historical STEMI/NSTEMI/UA independent multilabel feasibility
model: Xception + InceptionTime + Lead-GAT + fixed Morlet CWT-CNN + class-specific Noisy Top-2 + KAN
loss: fold-train weighted BCEWithLogits + record-level exclusivity + router balance - entropy
seed: 2026
fold_seed: 2026
device: NVIDIA RTX A6000 GPU1
---

> **任务定义更正（2026-08-30）**：本可行性实验使用旧的三个独立 sigmoid
> ACS 输出，不能代表当前互斥 ACS 亚型任务。当前正式协议为
> `OTHER/STEMI/NSTEMI/UA` 单标签四分类 + 独立 OMI 二分类；本报告结果仅作
> 工程历史记录。

# Revised sparse HMoE-KAN feasibility experiment

## 1. 结论先行

这次新方案在 GPU1 上完成了患者级五折交叉验证和 OOF 评估，代码、训练、路由、阈值和校准产物均已生成。模型可以稳定运行，但当前结果不足以证明新方案优于已有三专家 E4，也不建议现在把它作为论文核心创新点：

- OOF 宏 AUROC = **0.7698**，宏 AUPRC = **0.4254**；
- 记录级 NSTEMI-vs-UA 子集（n=7,448）的无阈值 AUROC = **0.7994**、AUPRC = **0.4502**；
- 在联合阈值 T4 下，NSTEMI sensitivity = **0.5117**、UA sensitivity = **0.8872**，但 NSTEMI sensitivity 仍偏低、UA specificity 仅 **0.2956**；
- 三个类别的路由分布几乎没有差异（NSTEMI/UA JS divergence = **0.00018**，cosine = **0.9993**），说明“类别专属路由”尚未学出明显的类别分工；
- 与此前三专家 WavelNet E4 报告的五折均值（宏 AUROC 约 0.7961、宏 AUPRC 约 0.4689、宏 F1 约 0.5109）相比，本次结果没有涨点。两次实验不是严格 paired ablation，因此只能作为工程可行性和风险信号，不能当作最终显著性结论。

**判断：No convincing improvement / 方案可运行但暂不 Go。** 新模型解决了“能否按 README 跑起来”的问题，但没有满足 README 中“异构专家、Wavelet、类别专属路由和 KAN 带来稳定收益”的 Go 条件。

## 2. 数据与划分审计

本次严格使用服务器现有 ACS 预处理数据，没有重新随机划分，也没有使用外部测试集标签：

| 项目 | 本次实际使用 |
|---|---:|
| 可见训练记录 | 17,960 |
| 可见训练患者 | 17,018 |
| 隐藏外部测试记录 | 1,995（未用于阈值或训练） |
| 输入形状 | `[N, 1000, 12]` |
| 采样率/时长 | 100 Hz / 10 s |
| STEMI 阳性 | 1,442 |
| NSTEMI 阳性 | 1,235 |
| UA 阳性 | 6,213 |
| all-zero 记录 | 9,070 |

README 中引用的 19,955 条、18,909 位患者、500 Hz 原始数据规格与当前服务器上的 P2 数组不完全一致。本次报告以当前实际可复现数组为准，不能把 README 的原始论文数据规格冒充为本地数据规格。

划分协议为 `patient-level joint-stratified 5-fold`，`fold_seed=2026`。同一患者不会同时进入 train 和 validation。每个 fold 仅用该 fold 的训练记录拟合 robust scaler 和 `pos_weight`；增强只在训练集执行，validation 不 oversampling、不增强。

标签审计需要区分两个层次：

1. **记录层面**：17,960 条记录只出现 `all-zero` 或单个 STEMI/NSTEMI/UA 阳性，三对标签共现均为 0，因此本次对每条 ECG 记录加入轻量 exclusivity loss 是有记录层依据的。
2. **患者层面**：同一患者的历史记录可能属于不同标签，患者 joint state 不是互斥的（state bitmask 3/5/6/7 分别有 8/47/58/5 位患者）。因此 exclusivity 只约束单条 ECG 记录的输出，不能解释为“患者终身诊断互斥”。

## 3. 实际实现的模型

四个专家都输出 128 维，再投影到 `D=256`：

1. **Xception morphology expert**：复用项目现有 XceptionTime 特征体；
2. **InceptionTime multi-scale expert**：复用项目现有 InceptionTime 特征体；
3. **Lead-GAT spatial expert**：每导联 CNN 编码后构建 12 节点完整图，使用 PyTorch Geometric `GATConv`；
4. **Morlet CWT-CNN frequency expert**：100 Hz 输入上用固定、可微的 Morlet 实部/虚部卷积核组得到 CWT magnitude，再接二维 CNN；没有 MVMD/VMD，也没有把 MVMD 当作网络外部预处理。

随后使用一个共享 ECG router stem，接三个独立的类别路由器。每个路由器在四个专家中加入训练期高斯噪声并选择 Top-2；选中的两个专家特征和路由分数进入类别专属 combiner，最后接三个独立 KAN head（`256 -> 64 -> 1`）。输出为三个 sigmoid logits，不使用 softmax，也不强制每条记录必须有一个阳性标签。

训练配置：Adam、学习率 `1e-4`、batch size `32`、最多 100 epochs、early stopping patience `15`、梯度裁剪 `1.0`、无 scheduler。损失为：

`weighted BCEWithLogitsLoss + 0.05 * record_exclusivity + 0.01 * router_balance - 0.001 * router_entropy`

正类权重只由每个 fold 的训练集计算，五折分别约为 STEMI `11.43–11.48`、NSTEMI `13.48–13.57`、UA `1.88–1.90`。

## 4. 五折训练记录

| Fold | Best epoch | Validation macro AUROC | Train time (s) |
|---:|---:|---:|---:|
| 0 | 13 | 0.7751 | 807.08 |
| 1 | 23 | 0.7837 | 1097.03 |
| 2 | 35 | 0.7776 | 1443.45 |
| 3 | 19 | 0.7876 | 984.60 |
| 4 | 31 | 0.7812 | 1332.50 |

每折可训练参数量均为 2,010,622。训练期间没有 NaN、OOM 或进程异常。

## 5. OOF 指标与阈值

所有阈值都在 OOF validation prediction 上确定；没有使用隐藏外部测试标签。T4 是在 T2 的基础上对 NSTEMI/UA 做联合网格搜索，并使用 normalized-margin 冲突消解。

| 策略 | STEMI/NSTEMI/UA threshold | Macro AUROC | Macro AUPRC | Macro F1 | Pairwise N/U AUROC | Pairwise N/U F1 | Sens NSTEMI | Sens UA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T0 fixed 0.5 | 0.50 / 0.50 / 0.50 | 0.7698 | 0.4254 | 0.4329 | 0.7994 | 0.6599 | 0.7304 | 0.7496 |
| T1 Youden | 0.3658 / 0.5102 / 0.4440 | 0.7698 | 0.4254 | 0.4295 | 0.7994 | 0.6651 | 0.7150 | 0.7628 |
| T2 max F1 | 0.87 / 0.76 / 0.27 | 0.7698 | 0.4254 | 0.4801 | 0.7994 | 0.6932 | 0.5117 | 0.8872 |
| **T4 joint NSTEMI/UA** | **0.87 / 0.76 / 0.27** | **0.7698** | **0.4254** | **0.4801** | **0.7994** | **0.6932** | **0.5117** | **0.8872** |
| T3 balanced accuracy | 0.36 / 0.51 / 0.44 | 0.7698 | 0.4254 | 0.4289 | 0.7994 | 0.6659 | 0.7142 | 0.7642 |

### T4 per-class metrics

| Label | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STEMI | 0.9092 | 0.6047 | 0.5895 | 0.9640 | 0.5886 | 0.9642 | 0.5891 | 0.5531 | 0.9340 |
| NSTEMI | 0.7765 | 0.2405 | 0.3514 | 0.9324 | 0.2775 | 0.9511 | 0.3101 | 0.2548 | 0.8925 |
| UA | 0.6238 | 0.4309 | 0.8646 | 0.2956 | 0.3937 | 0.8051 | 0.5410 | 0.1785 | 0.4925 |
| **Macro/overall** | **0.7698** | **0.4254** | — | — | — | — | **0.4801** | — | — |

T4 confusion matrices in `TP, TN, FP, FN` order：

- STEMI: `850, 15,924, 594, 592`;
- NSTEMI: `434, 15,595, 1,130, 801`;
- UA: `5,372, 3,473, 8,274, 841`.

NSTEMI/UA pairwise 子集（true rows = UA, NSTEMI；predicted columns = UA, NSTEMI）为：

```text
[[5512, 701],
 [ 603, 632]]
```

这说明 T4 虽然提高了 pairwise macro-F1 和 UA sensitivity，但代价是 NSTEMI sensitivity 降到 0.5117，且 UA 假阳性很多。若临床目标优先减少 NSTEMI 漏诊，T0/T1 的工作点反而更合适；最终阈值不能只按总 F1 选择。

## 6. 路由与校准诊断

OOF 平均 soft router probability：

| Class | Xception | Inception | Lead-GAT | Morlet CWT |
|---|---:|---:|---:|---:|
| STEMI | 0.2556 | 0.2531 | 0.3093 | 0.1821 |
| NSTEMI | 0.2516 | 0.2543 | 0.2388 | 0.2554 |
| UA | 0.2558 | 0.2540 | 0.2242 | 0.2661 |

STEMI 稍偏向 Lead-GAT，UA 稍偏向 CWT，但 NSTEMI 和 UA 的整体路由几乎相同（JS `0.00018`、cosine `0.99931`）。这不支持“类别专属路由已经形成明显分工”的强结论，也提示 balance/entropy 项可能把路由推得过于平均。

OOF calibration：STEMI Brier/ECE = `0.0911/0.1272`，NSTEMI = `0.1687/0.2609`，UA = `0.2288/0.0986`。尤其 NSTEMI 受 class-weighted BCE 影响存在明显过置信，后续若要提交外部平台或做临床解释，需在训练集内再切 calibration split 或使用温度/等距校准，且校准参数不能看测试标签。

## 7. 可行性判定与方案问题

### 已验证可行的部分

- 四专家图、时域、多尺度和 CWT 分支可以端到端前向/反向；
- 三个类别专属 Noisy Top-2 路由、KAN head 和多标签 sigmoid 输出可以稳定训练；
- 患者级五折没有泄漏，训练/验证预处理边界清楚；
- 阈值、pairwise NSTEMI/UA、confusion matrix、校准和路由产物均可自动生成。

### 当前没有通过的部分

- 新模型宏 AUROC/AUPRC/F1 低于此前三专家 E4 的报告均值，不能声称“加入 CWT、Inception 和 class-specific routing 后涨点”；
- NSTEMI 仍是主要薄弱类别（AUPRC `0.2405`、T4 F1 `0.3101`）；
- UA 的阈值工作点很容易以牺牲 specificity 换 sensitivity；
- NSTEMI/UA 路由分布接近相同，类别专属路由尚未提供可观测的专家选择差异；
- weighted BCE 造成 NSTEMI 概率过置信，当前模型不适合直接把 sigmoid 概率当作校准风险概率；
- 当前是内部 OOF 可行性验证，尚未对这套新模型生成外部平台提交或外部测试指标，因此不能把 OOF 数值称为外部测试性能。

### 后续建议（不改变本次结果）

1. 做严格 paired ablation：固定五折和 seed，逐项比较 `4 experts concat`、`soft gate`、`shared Top-2`、`class-specific Top-2`、`KAN vs MLP`、`CWT vs WavelNet`，不要同时改变 loss 和采样策略。
2. 单独消融 `exclusivity/balance/entropy`；当前记录层互斥成立，但 balance/entropy 可能抑制类别路由差异。
3. 以 NSTEMI sensitivity 或 pairwise balanced accuracy 作为验证选择约束，并把 T0/T1/T4 同时保留，不要只报告 T2/T4 的高宏 F1 工作点。
4. 在冻结架构和阈值后，再训练五折 ensemble 并提交外部平台；外部标签只能用于最后一次盲评，不能回流调参。
5. 对 NSTEMI 做校准和不确定性分析，再决定是否值得作为论文 loss/路由创新点。基于本次结果，论文表述应为“feasibility/ablation candidate”，而不是已证实的核心创新。

## 8. 产物位置

本地报告：

`reports/acs_sparse_hmoe_kan_feasibility_gpu1_20260826.md`

服务器完整结果目录：

`/data/code/ouyanghao/oyh/ACS-OMI-AMI/results/acs_sparse_hmoe_kan_fullcohort_gpu1_20260826/`

主要文件：

- `summary.json`：五折信息、阈值、全部 OOF 指标、路由和校准摘要；
- `metrics_by_threshold.json`、`thresholds.json`：T0–T4；
- `label_audit.json`、`label_cooccurrence.csv`、`split.json`：数据和划分审计；
- `oof_predictions.csv/.npz`：全量 OOF 标签/概率；
- `fold_0..4_checkpoint.pt`、`history_fold_0..4.json`：checkpoint 和训练曲线数据；
- `router_usage.csv`、`router_summary.json`：Top-2 选择和类别路由；
- `roc_oof.png`、`reliability_oof.png`、`confusion_matrix_t4.png`、`router_probability_heatmap.png`：图形产物；
- `run.log`：GPU1 完整训练日志。
