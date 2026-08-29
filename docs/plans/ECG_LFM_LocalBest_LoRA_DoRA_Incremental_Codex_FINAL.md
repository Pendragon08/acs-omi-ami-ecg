# ECG-LFM 微调策略对比实验方案（先扫描本地最佳方案，再做 LoRA/DoRA 增量优化）

> **任务定义更正（2026-08-29）**：本文件的早期章节把
> STEMI/NSTEMI/UA/OMI 写成四个独立 sigmoid，仅保留作历史方案和审计依据。
> 当前执行版使用 ACS `OTHER/STEMI/NSTEMI/UA` 单标签分类头（softmax/argmax）
> 与独立 OMI/NOMI 二分类头；修正版配置和结果见
> [`CORRECTED_TASK_FORMULATION.md`](CORRECTED_TASK_FORMULATION.md)、
> [`corrected_ecg_lfm_joint_20260829.md`](../reports/corrected_ecg_lfm_joint_20260829.md)。

> **用途**：本报告直接交给 Codex 执行。  
> **前置条件**：上一版 ECG-LFM 四标签方案已经在本地实现并完成实验，包括 raw 500 Hz WFDB 数据、ECG-LFM 原生预处理、四标签对齐、patient-level 5-fold、external test、官方 `pretrain.pt` 加载以及至少一种已验证的微调方案。  
> **本次第一步不是重新实现原论文 PFT**，而是要求 Codex **扫描当前本地项目，找出目前真实存在且结果最优的 ECG-LFM 微调方案**，完整记录它的代码、配置、冻结策略、训练参数、checkpoint 规则和指标，作为 `LOCAL_BEST_BASELINE`。  
> **本次任务属于增量优化**：禁止重复下载原始数据、禁止重新实现 ECG-LFM preprocessing、禁止重新划分 folds；必须在 `LOCAL_BEST_BASELINE` 的现有代码路径上新增 LoRA / DoRA。  
> **实验目标**：在完全相同的数据缓存、预处理、标签定义、fold、external test、loss/evaluation protocol 下，只改变参数适配策略，比较本地最佳基线与 LoRA / DoRA：
>
> 1. 原论文/官方基线：Partial Fine-tuning（PFT）
> 2. LoRA：Low-Rank Adaptation
> 3. DoRA：Weight-Decomposed Low-Rank Adaptation
>
> 核心问题不是“DoRA 是否新”，而是：
>
> **大型 ECG foundation model 应该如何有效迁移到样本规模较小、类别不平衡、且 NSTEMI/UA 边界细粒度的 ACS/OMI 四标签任务？**

---

# 0. 本次任务的第一原则：先扫描本地真实最佳方案

Codex 开始修改代码前，必须先对当前本地 ECG-LFM 工程进行一次完整审计。

**不要根据本 README 假定当前最佳方案一定是原论文 Partial Fine-tuning。**

Codex 必须从本地实际代码与实验结果中确认：

```text
当前 ECG-LFM 最优实验是哪一个？
```

建议搜索范围：

```text
ECG-LFM model wrappers
training scripts
experiment configs
fold configs
results / outputs
metrics json/csv
training logs
checkpoint metadata
threshold files
external-test summaries
README / notes
```

重点搜索关键词：

```text
ecg_lfm
ECGLFM
finetune
fine_tune
partial_ft
freeze
feature_grad_mult
pretrain.pt
fold
oof
threshold
external
macro_auroc
macro_auprc
best
```

如果当前本地已经存在多个 ECG-LFM 微调实验，例如：

```text
official-style partial FT
modified partial FT
full FT
different LR
different scheduler
different checkpoint criterion
different head
different loss
```

Codex 必须先汇总它们的已有结果，然后根据**当前项目已经采用的主评价标准**确定最佳方案。

若当前项目没有显式写“primary metric”，优先按：

```text
1. OOF Macro AUPRC
2. OOF Macro AUROC
3. NSTEMI AUPRC / F1
```

综合判断，并在报告里说明选择依据。

最终将真实最优方案命名为：

```text
LOCAL_BEST_BASELINE
```

后续 LoRA / DoRA 必须基于这个方案做增量修改，而不是基于假设中的原论文配置重新造一套 baseline。

---


# 1. Codex 本地扫描要求（修改代码前必须完成）

Codex 必须先扫描当前工作区，而不是从 README 假设目录。

建议执行思路：

```text
1. 找 ECG-LFM 相关代码
2. 找训练配置
3. 找最近/最佳实验输出
4. 追溯最佳结果对应的 config + checkpoint + commit/code path
5. 还原真实 fine-tuning strategy
6. 写 baseline snapshot
7. 再开始 LoRA / DoRA 修改
```

优先根据现有项目结构搜索，不要因为本文档示例路径与实际仓库不同就新建重复文件。

Codex 在开始修改前，应先向终端输出一份简短审计摘要：

```text
Detected local best ECG-LFM experiment:
- experiment:
- code path:
- config:
- preprocessing source:
- fold source:
- trainable modules:
- loss:
- optimizer / LR:
- scheduler:
- checkpoint rule:
- OOF Macro AUROC:
- OOF Macro AUPRC:
- NSTEMI AUPRC:
- external result:
- why selected as local best:
```

只有这一步完成后，才进入 adapter 实现。

---

# 2. 当前任务定义

数据集：

**A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome prediction containing 19,955 ECGs**

Scientific Data, 2026

论文：
https://www.nature.com/articles/s41597-026-07278-0

DOI：
`10.1038/s41597-026-07278-0`

官方数据：
https://doi.org/10.6084/m9.figshare.29925314

已确认数据规格：

```text
19,955 ECGs
18,909 patients
12 leads
500 Hz original sampling
10 s
WFDB .dat + .hea
```

四个目标标签固定为：

```text
STEMI
NSTEMI
UA
OMI
```

输出：

```text
4 independent logits
→ 4 independent sigmoid probabilities
```

不是四分类 Softmax。

---

# 3. 固定数据划分

必须直接读取上一版 ECG-LFM 已经固定并验证过的划分：

```text
Training pool:
17,960 ECGs
17,018 patients

External test:
1,995 ECGs
```

训练池内部：

```text
patient-level iterative multilabel stratified 5-fold
seed = 2026
```

要求：

```text
同一患者禁止跨 fold
External test 禁止用于：
- training
- validation
- checkpoint selection
- hyperparameter selection
- threshold tuning
```

PFT / LoRA / DoRA 必须直接复用上一版完全相同的：

```text
record IDs
patient IDs
fold assignments
external test
```

---

# 4. 直接复用上一版 ECG-LFM 数据与预处理结果

本次实验**不再进行任何原始数据下载或 ECG-LFM preprocessing 开发**。

上一版方案已经完成：

```text
Scientific Data 官方 raw 500 Hz WFDB ECG
        ↓
ECG-LFM 原生预处理
        ↓
1 Hz high-pass
30 Hz 2nd-order Butterworth low-pass
50/60 Hz notch
z-score normalization
        ↓
10 s → 2 × 5 s segments
        ↓
可直接输入 ECG-LFM 的预处理结果/缓存
```

本次 Codex 必须直接复用上一版已经验证通过的：

```text
raw 500 Hz 数据目录
record_id / patient_id 对齐表
STEMI / NSTEMI / UA / OMI 标签
train / external split
patient-level 5-fold assignment
ECG-LFM preprocessing code
ECG-LFM preprocessing cache / manifest
segment aggregation code
evaluation code
threshold code
external ensemble code
```

禁止：

```text
重新下载 Figshare 数据
重新生成一套 raw dataset
重新设计滤波流程
重新实现 z-score
重新切分 5 s segment
重新随机划分 folds
复制出第二套 ECG-LFM 数据 pipeline
```

如果现有 pipeline 已经有统一 experiment/config 接口：

> 本次只增加 `finetuning_method = partial_ft | lora | dora` 及相应 adapter 参数。

本次实验的变量必须严格限制为：

```text
参数更新方式
+
adapter 类型及其超参数
```

其余数据和 evaluation pipeline 全部继承上一版实现。

---


# 5. Foundation Model：ECG-LFM

主要论文：

**A self-supervised electrocardiogram foundation model for empowering cardiovascular disease prediction and genetic factor discovery**

Nature Communications, 2026

论文：
https://www.nature.com/articles/s41467-026-72436-2

官方代码：
https://github.com/biomed-AI/ECG-LFM

官方预训练权重：
https://zenodo.org/records/20388950

ECG-LFM 在：

```text
11,571,587 12-lead ECGs
```

上进行自监督预训练。

官方分类器路径：

```text
fairseq_signals/models/classification/ecg_transformer_classifier.py
```

官方 fine-tuning config：

```text
examples/ECGLFM/config/finetuning/diagnosis.yaml
```

---

# 6. 基线定义：LOCAL_BEST_BASELINE（以本地扫描结果为准）

原论文的 Partial Fine-tuning 只作为**方法参考**，不再预设为本次实验的最终 baseline。

Codex 必须先扫描并生成：

```text
artifacts/ecg_lfm_local_best_baseline.md
artifacts/ecg_lfm_local_best_baseline.json
```

至少记录：

```text
experiment_name
experiment_path
config_path
train_script
model_wrapper
checkpoint_path / checkpoint rule
ECG-LFM commit / model version
pretrained checkpoint path
preprocessing cache path

feature extractor frozen?
context encoder frozen/trainable?
which layers are trainable?
classification head architecture

loss
optimizer
learning rate
scheduler
batch size
gradient accumulation
max epochs
early stopping
random seed
checkpoint selection metric

5-fold split source
OOF aggregation
threshold strategy
external ensemble strategy

OOF Macro AUROC
OOF Macro AUPRC
OOF Macro F1
per-label AUROC/AUPRC/F1
NSTEMI-vs-UA metrics
external metrics（若已有）
```

同时输出实际 trainable parameter names：

```text
artifacts/local_best_trainable_params.txt
```

以及参数量：

```text
total_params
trainable_params
trainable_ratio
```

## 5.1 如果本地最佳方案就是原论文式 PFT

则直接把它作为：

```text
LOCAL_BEST_BASELINE = Partial FT
```

不需要重新实现一次。

## 5.2 如果本地最佳方案已经做过优化

例如：

```text
不同 LR
不同 scheduler
不同 checkpoint selection
不同 unfreeze 策略
不同 classification head
```

则必须保留这些已经验证有效的设置。

LoRA / DoRA 对比时：

```text
保留本地最佳方案的数据、head、loss、训练流程、evaluation protocol
只替换 backbone adaptation strategy
```

## 5.3 原论文 PFT 的作用

原论文 PFT 继续作为文献方法依据，用来说明 ECG-LFM 官方下游迁移方式。

但本次实验主比较应写成：

```text
LOCAL_BEST_BASELINE
vs
LOCAL_BEST + LoRA adaptation
vs
LOCAL_BEST + DoRA adaptation
```

而不是机械地：

```text
paper PFT
vs
LoRA
vs
DoRA
```

---


# 7. LoRA 方法依据

原始论文：

**LoRA: Low-Rank Adaptation of Large Language Models**

ICLR 2022

OpenReview：
https://openreview.net/forum?id=nZeVKeeFYf9

核心思想：

冻结原始权重 `W0`，只学习低秩增量：

```math
W' = W0 + ΔW
```

其中：

```math
ΔW = (α / r) BA
```

且：

```math
r << d
```

---

# 8. ECG 场景中的 LoRA 直接依据

可作为 ECG-specific supporting reference：

**Toward robust automated cardiovascular arrhythmia detection using self-supervised learning and 1-dimensional vision transformers**

Scientific Reports, 2026

论文：
https://www.nature.com/articles/s41598-026-41549-5

代码：
https://github.com/Mitchell-Chatterjee/Robust-Automated-Cardiovascular-Arrhythmia-Detection

该工作明确在 ECG Transformer fine-tuning 中使用 LoRA，并报告：

```text
rank-8 LoRA
```

同时强调，对 ECG Transformer：

```text
将 LoRA 覆盖到主要 linear layers
```

比只改极少 attention projection 更合理。

注意：

> 这篇论文用于证明 LoRA 在 ECG Transformer 场景中已经可行，不意味着其具体 architecture 或参数必须照搬。

---

# 9. DoRA 方法依据

论文：

**DoRA: Weight-Decomposed Low-Rank Adaptation**

ICML 2024 Oral

ICML：
https://icml.cc/virtual/2024/poster/35052

官方代码：
https://github.com/NVlabs/DoRA

DoRA 的目标：

```text
让参数高效微调的更新行为更接近 Full Fine-tuning
```

基本分解：

```math
W = m * V / ||V||
```

其中：

```text
m = magnitude
V/||V|| = direction
```

DoRA 对 direction 使用 LoRA：

```math
V = W0 + (α / r) BA
```

训练：

```text
W0         frozen
A/B        trainable
magnitude  trainable
```

因此 DoRA 比 LoRA 多学习：

```text
weight magnitude
```

---

# 10. DoRA 实现参考

Hugging Face PEFT 已支持：

```python
LoraConfig(
    use_dora=True,
    ...
)
```

官方说明：
https://huggingface.co/docs/peft/package_reference/lora_variant_dora

但 ECG-LFM 是：

```text
fairseq_signals custom PyTorch model
```

不是标准 Hugging Face Transformer。

因此 Codex 必须：

```text
先尝试 PEFT 对当前 nn.Linear 的兼容性
```

如果无法稳定 wrap：

```text
实现自定义 DoRALinear wrapper
```

禁止为了使用 PEFT 而重写 ECG-LFM backbone。

---

# 11. 本实验核心对比

**实际基线由本地扫描确定。**

主实验定义：

| ID | Method | Base pipeline | Backbone adaptation | Classifier / loss / evaluation |
|---|---|---|---|---|
| FT-B | **LOCAL_BEST_BASELINE** | 当前本地最优 ECG-LFM 方案 | 保持现状 | 保持现状 |
| FT-L | **LOCAL_BEST + LoRA** | 与 FT-B 相同 | 用 LoRA 替换对应 trainable context adaptation | 与 FT-B 相同 |
| FT-D | **LOCAL_BEST + DoRA** | 与 FT-B 相同 | 用 DoRA 替换对应 trainable context adaptation | 与 FT-B 相同 |

核心公平原则：

```text
FT-B / FT-L / FT-D
```

除参数适配方式外，尽量全部复用同一条本地 pipeline。

如果 `LOCAL_BEST_BASELINE` 自身不是标准 Partial FT，也不要强行改回论文配置。

原论文 Partial FT 仅作为：

```text
methodological reference
```

若本地已有其结果，可作为附加对比；没有必要为了本实验重复跑一次。

---


# 12. LoRA 主实验结构

```text
ECG-LFM pretrained model
        │
        ├── Feature Extractor
        │     Frozen
        │
        └── Context Encoder
              Base weights Frozen
                    │
              LoRA adapters
                    │
              ECG representation
                    │
              Mean Pooling
                    │
              Linear Head
                    │
           STEMI/NSTEMI/UA/OMI
```

真正更新：

```text
LoRA A
LoRA B
4-label classification head
```

其余参数全部 frozen。

---

# 13. DoRA 主实验结构

```text
ECG-LFM pretrained model
        │
        ├── Feature Extractor
        │     Frozen
        │
        └── Context Encoder
              Base weights Frozen
                    │
              DoRA adapters
             /            \
      low-rank direction   magnitude
             \            /
              ECG representation
                    │
              Mean Pooling
                    │
              Linear Head
                    │
           STEMI/NSTEMI/UA/OMI
```

真正更新：

```text
DoRA low-rank A/B
DoRA magnitude
4-label classification head
```

---

# 14. Adapter 应注入哪些层

Codex 不要先猜。

必须先：

```python
for name, module in model.named_modules():
    print(name, type(module))
```

确认 `pretrain.pt` 实际使用的 encoder 路径。

---

# 15. Transformer 路径推荐 target

如果实际 Context Encoder 使用标准 Transformer layer，优先：

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.out_proj

fc1
fc2
```

即：

```text
Attention + FFN All-Linear
```

LoRA 和 DoRA 必须使用相同 target modules。

---

# 16. Conformer 路径推荐 target

如果实际 ECG-LFM 使用 Conformer：

优先适配：

```text
Self-attention projection linear layers

ffn1.w_1
ffn1.w_2

ffn2.w_1
ffn2.w_2
```

第一阶段不要适配：

```text
depthwise convolution
pointwise convolution
LayerNorm
BatchNorm
positional encoding
```

原因：

> 主实验要比较 low-rank linear adaptation，而不是同时改变 convolutional adaptation。

---

# 17. 为什么使用 All-Linear Adapter

第一版不推荐：

```text
q_proj + v_proj only
```

而推荐：

```text
Attention projections
+
FFN linear layers
```

原因：

```text
ACS/OMI 是细粒度临床 domain adaptation
```

不仅需要改变 attention pattern，还可能需要改变高层 feature transformation。

同时已有 2026 ECG Transformer 工作支持 ECG 场景中覆盖主要 Linear 层进行 LoRA adaptation。

因此主方案命名：

```text
ECG-LFM-LoRA-AllLinear
ECG-LFM-DoRA-AllLinear
```

---

# 18. 主 Adapter 超参数

为了先做一个干净的主实验：

## LoRA

```text
rank r        = 8
alpha         = 16
dropout       = 0.05
LR            = 1e-4
```

## DoRA

```text
rank r        = 8
alpha         = 16
dropout       = 0.05
LR            = 5e-5
```

DoRA 第一版先使用稍低于 LoRA 的 LR。

---

# 19. 超参数搜索原则

主论文不能为了找到最好结果对 external test 反复调。

第一阶段固定：

```text
LoRA:
r=8, alpha=16, dropout=0.05, lr=1e-4

DoRA:
r=8, alpha=16, dropout=0.05, lr=5e-5
```

如果需要第二阶段 rank sensitivity：

```text
r = 4
r = 8
r = 16
```

只使用 training/validation folds。

External test 在所有方案锁定后只进行最终评估。

---

# 20. Loss 必须完全一致

三个主实验均使用：

```text
unweighted BCEWithLogitsLoss
```

禁止在 LoRA / DoRA 中额外引入：

```text
Focal Loss
Asymmetric Loss
pos_weight
oversampling
WeightedRandomSampler
```

否则无法判断性能变化来自：

```text
fine-tuning strategy
```

还是：

```text
loss / sampling
```

---

# 21. 两个 5 秒 segment 的 record-level 聚合

一条 10 秒 ECG：

```text
segment A → z_A ∈ R^4
segment B → z_B ∈ R^4
```

主方案：

```math
z_record = (z_A + z_B) / 2
```

然后：

```math
p_record = sigmoid(z_record)
```

所有：

```text
AUROC
AUPRC
F1
threshold
external prediction
```

都按原始 10 秒 ECG record 评价。

---

# 22. 五折训练流程

每个方法独立训练 5 folds。

例如 DoRA：

```text
Fold 1:
pretrain.pt
→ fresh DoRA adapter
→ fresh 4-label head
→ train
→ best checkpoint

Fold 2:
pretrain.pt
→ fresh DoRA adapter
→ fresh head
...

Fold 5
```

LoRA 同理。

每 fold 必须独立初始化。

---

# 23. Checkpoint selection

三种方法统一：

```text
validation loss
```

Early stopping：

```text
patience = 10
```

Max epochs：

```text
100
```

---

# 24. Batch size

目标：

```text
effective batch size = 256
```

如果 GPU 不够：

```text
physical batch = 16/32/64/128
+
gradient accumulation
```

必须记录：

```text
physical batch
accumulation steps
effective batch
```

---

# 25. OOF Threshold

每个方法必须有自己的：

```text
5-fold OOF probabilities
```

然后分别：

```text
STEMI
NSTEMI
UA
OMI
```

在：

```text
0.01–0.99
```

网格中搜索 Youden J。

得到：

```text
PFT thresholds
LoRA thresholds
DoRA thresholds
```

三组阈值独立。

禁止互相复用。

---

# 26. External Test

每个方法：

```text
5 fold models
→ 5 probabilities
→ mean ensemble
→ method-specific frozen OOF thresholds
→ external test results
```

External：

```text
1,995 ECG
```

标签只能在最终 evaluation 时使用。

---

# 27. Primary Metrics

主指标：

```text
Macro AUROC
Macro AUPRC
```

考虑数据不平衡：

> AUPRC 必须与 AUROC 同等重视。

---

# 28. Per-label Metrics

每个标签：

```text
AUROC
AUPRC
F1
Sensitivity
Specificity
PPV
NPV
```

特别关注：

```text
NSTEMI AUPRC
NSTEMI F1
OMI AUPRC
```

---

# 29. NSTEMI vs UA 专项分析

从 ground truth 中筛：

```text
NSTEMI positive
or
UA positive
```

计算：

```text
NSTEMI vs UA AUROC
NSTEMI vs UA AUPRC
Balanced Accuracy
Macro F1
NSTEMI sensitivity
UA sensitivity
confusion matrix
```

---

# 30. Efficiency Metrics

必须记录：

```text
Total parameters
Trainable parameters
Trainable parameter ratio
Peak GPU memory
Training time / fold
Total 5-fold training time
Inference time / ECG
Checkpoint size
Adapter size
```

主表：

| Method | Trainable Params | Trainable % | Peak VRAM | Train Time | Macro AUROC | Macro AUPRC | NSTEMI AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| PFT | | | | | | | |
| LoRA | | | | | | | |
| DoRA | | | | | | | |

---

# 31. 参数统计必须真实计算

Codex 添加：

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)
```

同时导出 trainable parameter names。

需要确认：

## LoRA

```text
只有：
adapter A/B
classifier
```

可训练。

## DoRA

```text
只有：
A/B
magnitude
classifier
```

可训练。

如果发现大量 backbone 参数：

```text
requires_grad=True
```

立即停止训练并修复。

---

# 32. Adapter 参数保存

推荐：

```text
base:
shared pretrain.pt

fold checkpoints:
adapter weights
classification head
adapter config
```

例如：

```text
outputs/
  lora/
    fold0/
      adapter.pt
      head.pt
      config.json
  dora/
    fold0/
      adapter.pt
      head.pt
      config.json
```

---

# 33. PEFT 兼容性检查

因为 ECG-LFM 是 fairseq_signals：

Codex 首先尝试：

```text
Hugging Face PEFT wrap custom torch model
```

必须做 smoke test：

```text
1. target modules 正确找到
2. adapter 参数注入成功
3. base weights frozen
4. forward 正常
5. backward 只有 adapter/head 有梯度
6. save/load adapter 后 prediction 一致
```

如果失败：

```text
不要重构 ECG-LFM 成 Hugging Face Transformer
```

改为：

```text
custom LoRALinear
custom DoRALinear
```

直接替换目标 `nn.Linear`。

---

# 34. Unit Tests

至少增加：

```text
test_adapter_injection
test_backbone_frozen
test_lora_trainables
test_dora_trainables
test_output_shape
test_adapter_save_load
test_split_leakage
```

输入：

```text
[B,12,2500]
```

输出：

```text
[B,4]
```

---

# 35. 第一阶段运行顺序

不要一开始跑完整 15 个 fold jobs。

```text
Stage 1  复现 PFT Fold 0
Stage 2  LoRA single-batch smoke test
Stage 3  LoRA small-subset overfit
Stage 4  LoRA Fold 0
Stage 5  DoRA single-batch smoke test
Stage 6  DoRA small-subset overfit
Stage 7  DoRA Fold 0
Stage 8  完整 PFT / LoRA / DoRA 5-fold
```

---

# 36. 本次应在上一版代码上增量扩展

优先复用现有配置，例如上一版已有：

```text
configs/ecg_lfm/finetune.yaml
```

则不要复制一个新的完整训练框架。

建议只增加类似：

```yaml
finetuning:
  method: partial_ft   # partial_ft | lora | dora

  lora:
    rank: 8
    alpha: 16
    dropout: 0.05
    target: all_linear_context

  dora:
    rank: 8
    alpha: 16
    dropout: 0.05
    target: all_linear_context
```

训练入口保持同一个：

```text
train_ecg_lfm(...)
```

内部根据：

```text
finetuning.method
```

决定：

```text
partial_ft
→ 原上一版行为

lora
→ freeze base + inject LoRA

dora
→ freeze base + inject DoRA
```

这样可以最大程度保证：

```text
数据
训练循环
指标
OOF
threshold
external ensemble
```

完全一致。

---

# 37. 上一版方案必须作为回归基线

新增：

```text
configs/ecg_lfm/
  partial_ft.yaml
  lora.yaml
  dora.yaml
```

LoRA：

```yaml
method: lora

adapter:
  rank: 8
  alpha: 16
  dropout: 0.05
  target: all_linear_context

optimizer:
  name: adam
  lr: 1.0e-4

loss:
  name: bce_logits
  weighted: false
```

DoRA：

```yaml
method: dora

adapter:
  rank: 8
  alpha: 16
  dropout: 0.05
  target: all_linear_context

optimizer:
  name: adam
  lr: 5.0e-5

loss:
  name: bce_logits
  weighted: false
```

---

# 38. 上一版方案必须作为回归基线

本次修改前，上一版 ECG-LFM Partial FT 已经是可运行的新方案。

Codex 修改完成后首先必须做 regression check：

```text
same fold
same seed
same checkpoint
same preprocessing cache
same PFT config
```

确认：

```text
修改前 PFT
≈
修改后 PFT
```

至少比较：

```text
validation loss
Macro AUROC
Macro AUPRC
per-label predictions
```

允许正常浮点/随机误差，但如果 PFT 指标发生明显变化：

```text
先停止 LoRA / DoRA 完整训练
```

优先排查本次 adapter 接入是否改变了：

```text
forward
pooling
segment aggregation
label order
loss
optimizer
evaluation
```

本实验的可信度依赖于：

> **PFT baseline 在新增 LoRA / DoRA 后仍然保持上一版行为不变。**

---

# 39. Fair Comparison 原则

允许变化：

```text
哪些参数 trainable
adapter type
method-specific LR
```

必须固定：

```text
raw ECG source
preprocessing
record set
folds
external set
labels
loss
segment aggregation
max epochs
early stopping
checkpoint rule
metrics
threshold protocol
ensemble protocol
random seed policy
```

---

# 40. Statistical Comparison

至少输出：

```text
5 fold metric mean ± std
```

推荐增加 OOF patient-level bootstrap：

```text
1000 bootstrap samples
```

比较：

```text
DoRA - PFT
LoRA - PFT
DoRA - LoRA
```

对于：

```text
Macro AUPRC
NSTEMI AUPRC
NSTEMI-vs-UA AUROC
```

给：

```text
95% CI
```

如果 CI 明显跨 0：

不要声称“显著提升”。

---

# 41. 论文 Contribution 推荐写法

不要写：

> We applied DoRA to ECG-LFM.

建议写成：

### Contribution 1：Foundation-model transfer

> We adapt a large-scale self-supervised ECG foundation model to simultaneous prediction of STEMI, NSTEMI, unstable angina, and occlusion myocardial infarction in an angiography-linked ACS cohort.

### Contribution 2：Adaptation strategy study

> We systematically compare conventional partial fine-tuning with low-rank and weight-decomposed parameter-efficient adaptation under identical patient-level cross-validation.

### Contribution 3：Fine-grained + efficiency evaluation

> We evaluate not only overall discrimination but also minority-class and NSTEMI–UA discrimination together with trainable parameters, memory consumption, training cost, and inference efficiency.

---

# 42. 如果 DoRA 最好，可强化的方法 Contribution

只有实验支持后才能写：

> **We show that weight-decomposed parameter-efficient adaptation provides a better performance–efficiency trade-off than conventional partial fine-tuning and standard LoRA for transferring ECG-LFM to fine-grained ACS/OMI prediction.**

如果 NSTEMI 特别明显：

> **The benefit is particularly evident for the label-limited NSTEMI task and NSTEMI–UA discrimination.**

没有数据支持前禁止写：

```text
DoRA significantly improves...
```

---

# 43. Recommended Experiment Table

| Model | Method | Trainable % | Macro AUROC | Macro AUPRC | STEMI AUPRC | NSTEMI AUPRC | UA AUPRC | OMI AUPRC | NSTEMI-UA AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ECG-LFM | Partial FT | | | | | | | | |
| ECG-LFM | LoRA | | | | | | | | |
| ECG-LFM | DoRA | | | | | | | | |

效率表：

| Method | Trainable Params | Peak VRAM | Train Time/Fold | Adapter Size | Inference Time |
|---|---:|---:|---:|---:|---:|
| PFT | | | | | |
| LoRA | | | | | |
| DoRA | | | | | |

---

# 44. Optional Ablation

主实验完成后再考虑：

```text
Rank:
r = 4 / 8 / 16

Target:
Attention-only
vs
All-Linear

Optional upper-cost baseline:
Full Fine-tuning
```

不要第一轮全部同时跑。

---

# 45. 不要在本实验同时做的事情

禁止第一轮同时增加：

```text
KAN head
GAT
WavelNet
MoE
culprit-vessel auxiliary task
Focal loss
ASL
pos_weight
oversampling
knowledge distillation
```

本实验必须保持：

```text
ECG-LFM backbone fixed
→ only adaptation strategy changes
```

---

# 46. Codex 最终执行清单

本次第一阶段是**扫描本地当前最优 ECG-LFM 方案**，不是写新代码。

```text
[ ] 1. 扫描当前项目所有 ECG-LFM 训练入口、配置和历史结果
[ ] 2. 找到已经完成的最佳 ECG-LFM 微调实验
[ ] 3. 明确“最佳”的评价依据，并记录已有 OOF / external 指标
[ ] 4. 生成 ecg_lfm_local_best_baseline.md/json
[ ] 5. 导出当前最佳方案的 trainable parameter names 和参数量
[ ] 6. 找到该最佳方案实际使用的 model wrapper / trainer / evaluator
[ ] 7. 找到已完成的 500 Hz preprocessing cache / manifest，仅复用
[ ] 8. 找到固定 patient-level 5-fold / external split，仅复用
[ ] 9. 对 LOCAL_BEST_BASELINE 做一次最小 regression/smoke check，确认可复现
[ ] 10. inspect 当前实际 ECG-LFM named_modules()
[ ] 11. 确认实际 Transformer / Conformer target Linear modules
[ ] 12. 在 LOCAL_BEST_BASELINE 现有代码路径增加 finetuning_method 配置
[ ] 13. 实现 LoRA-AllLinear
[ ] 14. 实现 DoRA-AllLinear
[ ] 15. 确认 LoRA / DoRA 没有改变 preprocessing/head/loss/evaluator
[ ] 16. 确认 LoRA 仅 adapter A/B + baseline head 等必要模块可训练
[ ] 17. 确认 DoRA 仅 A/B + magnitude + baseline head 等必要模块可训练
[ ] 18. adapter save/load smoke test
[ ] 19. LoRA / DoRA small-subset overfit
[ ] 20. LOCAL_BEST / LoRA / DoRA Fold 0 对比
[ ] 21. Fold 0 正常后再执行完整 5-fold
[ ] 22. 复用现有 OOF / threshold / external ensemble pipeline
[ ] 23. 每种 adaptation 方法独立生成阈值
[ ] 24. 输出 performance / NSTEMI-vs-UA / efficiency comparison
```

禁止：

```text
重新下载 raw ECG
重新实现 ECG-LFM native preprocessing
重新划分 folds
重新构造 external set
重新实现一个“原论文 PFT baseline”覆盖当前本地最佳方案
为了 PEFT 重写 ECG-LFM backbone
```

本次修改必须是：

```text
CURRENT LOCAL BEST ECG-LFM
          ↓
识别真实实现
          ↓
保留所有已验证配置
          ↓
只新增 LoRA / DoRA adaptation
          ↓
公平比较
```

---

# 47. Codex 必须提交的文件
# 48. Codex 必须提交的文件

至少：

```text
finetuning_comparison_summary.csv
finetuning_efficiency_summary.csv

local_best_oof.csv
lora_oof.csv
dora_oof.csv

local_best_thresholds.json
lora_thresholds.json
dora_thresholds.json

local_best_external.csv
lora_external.csv
dora_external.csv

nstemi_vs_ua_comparison.csv

adapter_target_modules.txt
trainable_params_local_best.txt
trainable_params_lora.txt
trainable_params_dora.txt
```

---

# 49. 参考文献与代码地址

## ECG-LFM

Paper:
https://www.nature.com/articles/s41467-026-72436-2

Code:
https://github.com/biomed-AI/ECG-LFM

Checkpoint:
https://zenodo.org/records/20388950

## ACS Dataset

Paper:
https://www.nature.com/articles/s41597-026-07278-0

Dataset:
https://doi.org/10.6084/m9.figshare.29925314

## LoRA

Paper:
https://openreview.net/forum?id=nZeVKeeFYf9

## DoRA

ICML:
https://icml.cc/virtual/2024/poster/35052

Official code:
https://github.com/NVlabs/DoRA

PEFT docs:
https://huggingface.co/docs/peft/package_reference/lora_variant_dora

## ECG-specific LoRA evidence

Paper:
https://www.nature.com/articles/s41598-026-41549-5

Code:
https://github.com/Mitchell-Chatterjee/Robust-Automated-Cardiovascular-Arrhythmia-Detection

---

# 50. 一句话实验定义

> **先扫描并锁定当前本地已经实现的 ECG-LFM 最佳微调方案作为 `LOCAL_BEST_BASELINE`，随后在完全复用其数据、500 Hz preprocessing、fold、head、loss、OOF、threshold 和 external evaluation pipeline 的前提下，仅将参数适配方式替换为 LoRA 和 DoRA，比较三种策略在 ACS/OMI 四标签任务中的性能、少数类泛化能力与训练效率。**


---

# 51. 最重要的研究问题

最终论文要回答的是：

```text
在当前本地已经优化到最好的 ECG-LFM 下游方案基础上，
能否通过 LoRA / DoRA 这种参数高效适配方式：

1. 保持或提升四标签总体性能；
2. 改善 NSTEMI 与 NSTEMI-vs-UA 细粒度区分；
3. 显著减少 trainable parameters / VRAM / checkpoint size；
4. 获得优于当前本地最佳微调策略的 performance–efficiency trade-off？
```

因此本次研究不是“重新复现 ECG-LFM 原论文”，而是：

> **以当前本地最佳 ECG-LFM 为真实工程和实验基线，研究更先进的 foundation-model adaptation strategy 是否还能进一步优化。**
