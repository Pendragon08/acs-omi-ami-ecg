# ACS-MAGPIE-MoE：STEMI / NSTEMI / UA 三标签 ECG Sparse MoE 方案（LEGACY—互斥亚型）

> **任务定义更正（2026-08-29）**：STEMI/NSTEMI/UA 必须作为单标签 ACS 分类头（含 OTHER），不能作为三个独立 sigmoid。本文保留为历史方案；当前联合 ACS+OMI 协议见 `CORRECTED_TASK_FORMULATION.md`。

> **用途**：本 README 用于指导 Codex 基于现有旧 MoE / XceptionNet 项目实现新方案，并作为后续论文 Methods、Ablation、Related Work 的设计底稿。  
> **数据集**：Du et al., *Scientific Data*, 2026, *A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome prediction containing 19,955 ECGs*.  
> DOI: `10.1038/s41597-026-07278-0`  
> **主任务**：12 导联 ECG → `STEMI / NSTEMI / UA` 三标签预测。  
> **临床重点子任务**：`NSTEMI vs UA`。  
> **主要参考框架**：MAGPIE, *npj Digital Medicine*, 2026，采用 `Sparse MoE + 8 Experts + Top-2 Routing + Load-balancing Loss`。  
> **本版原则**：先**直接迁移 MAGPIE 的 Sparse MoE 核心机制**，不再把 Xception / GAT / Wavelet / InceptionTime 全部作为异构专家。  
> **创新验证**：严格比较 `MLP/FFN Experts` 与 `KAN Experts`。只有 KAN 版本五折稳定更优，才把 “KAN Experts” 写成创新点。

---

# 1. 为什么这版方案要收敛成 MAGPIE-style Sparse MoE

之前的方案同时包含：

```text
Xception
+ InceptionTime
+ GAT
+ Wavelet
+ Sparse Router
+ KAN
```

虽然功能丰富，但存在明显的论文风险：

1. 模块太多，难以判断涨点来自哪里；
2. 每个 Expert 结构不同，参数量和计算量难以公平比较；
3. 如果最终结果提高，难以证明 Sparse MoE 本身有效；
4. KAN 是否有效也会被 GAT / Wavelet 等模块干扰；
5. 容易被审稿人评价为 module stacking。

因此新方案改为：

```text
先直接迁移正式高影响因子期刊 MAGPIE 的核心 Sparse MoE 机制
↓
在 ECG + ACS 任务上验证
↓
再只改变一个变量：MLP Expert → KAN Expert
```

这样实验逻辑最干净。

---

# 2. 主参考论文：MAGPIE

**Xie M, Xiao Q, Wu H, et al.**  
*Masked autoencoding, generalizable pretraining, and integrated experts for enhanced glioma segmentation.*  
**npj Digital Medicine**, 2026, 9:163.  
DOI: `10.1038/s41746-026-02347-5`

## 2.1 为什么它可以作为本研究主参考

MAGPIE 是正式同行评审的 Nature Portfolio 医学 AI 期刊论文。

其 Sparse MoE 核心设计：

```text
Input Tokens
    ↓
Router
    ↓
scores for 8 experts
    ↓
Top-2 selection
    ↓
only selected 2 FFN experts are activated
    ↓
weighted aggregation
    ↓
output
```

其核心组件包括：

```text
8 FFN Experts
Top-2 Sparse Routing
Load-balancing Loss
Top-K Ablation
Expert Utilization Analysis
```

论文还比较：

```text
Top-1
Top-2
Top-3
Dense activation
```

因此，我们可以直接迁移：

```text
8 Experts
+ Single Router
+ Top-2
+ sparse activation
+ load balancing
```

而不是重新发明一套 Router。

---

# 3. MAGPIE 到 ECG 的迁移原则

MAGPIE 原始场景：

```text
medical image tokens
→ sparse MoE
→ specialized FFN experts
```

本研究迁移为：

```text
12-lead ECG
→ XceptionNet shared encoder
→ temporal ECG tokens / feature tokens
→ MAGPIE-style sparse MoE
→ STEMI / NSTEMI / UA prediction
```

关键点：

> **XceptionNet 不是 Expert。**

XceptionNet 是统一的 ECG 特征提取 Backbone。

MoE Expert 位于 Xception 特征之后。

这样最接近 MAGPIE：

```text
shared encoder
→ sparse FFN expert layer
```

---

# 4. 总体模型架构

推荐主模型名：

```text
ACS-MAGPIE-MoE
```

结构：

```text
                           12-lead ECG
                      [B, 12, 5000]
                              │
                              ▼
                    1D XceptionNet Encoder
                    remove original classifier
                              │
                              ▼
                   Temporal Feature Map H
                      [B, L, D]
                              │
                              ▼
                       LayerNorm
                              │
                              ▼
                        One Router
                              │
                  scores for 8 Experts
                              │
                     Noisy Top-2 Train
                  Deterministic Top-2 Eval
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
      Selected Expert i                 Selected Expert j
          MLP/FFN                           MLP/FFN
             │                                 │
             └──────── weighted fusion ────────┘
                              │
                              ▼
                         Residual Add
                              │
                              ▼
                     ECG Token Representation
                              │
                              ▼
                   Global / Attention Pooling
                              │
                              ▼
                         Shared ECG Vector
                              │
                              ▼
                         Linear Head
                              │
                    3 logits / probabilities
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
       STEMI               NSTEMI                 UA
```

---

# 5. 为什么 Router 只有一个

MAGPIE 本身采用：

```text
one Router per MoE layer
```

Router 对 8 个 Expert 一次性打分：

```math
s = Router(h) ∈ R^8
```

然后：

```math
S(h) = TopK(s, 2)
```

因此：

```text
1 Router
→ 8 Expert scores
→ select Top-2
```

不是：

```text
8 Experts
→ 8 Routers
```

也不是：

```text
STEMI Router
NSTEMI Router
UA Router
```

本版为了最大程度直接迁移 MAGPIE，**主模型只使用一个 Router**。

类别特异 Router 不再放在主模型中。

如果以后需要扩展，可以做额外实验，但不作为第一篇模型的核心设计。

---

# 6. 为什么使用 8 个 Experts

MAGPIE 使用 8 个 Expert，不是因为有 8 种疾病。

8 个 Expert 都是结构相同、参数不同的 FFN。

```text
E1 = FFN
E2 = FFN
...
E8 = FFN
```

训练前：

```text
Expert 1 ≠ 某个固定临床模式
Expert 2 ≠ 某个固定临床模式
```

通过：

```text
Router
+ Top-2 selection
+ gradient optimization
```

让专家自动产生 specialization。

本研究第一版也直接采用：

```text
num_experts = 8
top_k = 2
```

然后做消融：

```text
4 experts
8 experts
12 experts（仅资源允许时）
```

不预先声称 8 是 ECG 最优数量。

---

# 7. XceptionNet 输出不要直接 GAP 后再路由

为了更接近 MAGPIE 的 token-level routing，推荐从 XceptionNet 的最后一个卷积 stage 获取 feature map：

```text
[B, C, L]
```

转置：

```text
[B, L, C]
```

再 projection 到：

```text
[B, L, D]
```

其中每个：

```text
h_t ∈ R^D
```

可以理解为一个 ECG temporal token。

Router 对每个 token 分别决定 Top-2 Experts：

```math
S(h_t) = Top2(Router(h_t))
```

这样不同 ECG 时间区域可以调用不同专家。

比：

```text
GAP → one global vector → one route
```

更接近 MAGPIE，也更能体现 Sparse MoE 的 token specialization。

---

# 8. MLP / FFN Expert 定义

## 8.1 Baseline Experts

主参考 MAGPIE 版本使用 8 个同构 FFN Experts。

建议：

```text
Input D
→ Linear(D, 4D)
→ GELU
→ Dropout
→ Linear(4D, D)
```

形式：

```math
E_i(h) =
W_{i2} GELU(W_{i1} h + b_{i1}) + b_{i2}
```

其中：

```text
i = 1 ... 8
```

所有 Expert 结构相同，但参数独立：

```math
θ_i != θ_j
```

---

# 9. Router 定义

## 9.1 Router logits

```math
s_t = W_r h_t + b_r
```

其中：

```text
s_t ∈ R^8
```

## 9.2 训练阶段

使用：

```text
Noisy Top-2
```

例如：

```math
s'_t = s_t + noise
```

然后：

```math
S_t = Top2(s'_t)
```

## 9.3 验证 / 测试

关闭 routing noise：

```math
S_t = Top2(s_t)
```

---

# 10. 被选 Expert 如何融合

对于 token `h_t`：

```text
Router scores
→ select i, j
```

对 Top-2 score 重新归一化：

```math
α_i =
exp(s_i) / [exp(s_i) + exp(s_j)]

α_j =
exp(s_j) / [exp(s_i) + exp(s_j)]
```

输出：

```math
MoE(h_t) =
α_i E_i(h_t)
+
α_j E_j(h_t)
```

再做 residual：

```math
h'_t = h_t + MoE(LN(h_t))
```

这一点直接参考 MAGPIE-style Sparse MoE。

本版**不使用异构 Expert concat + Combiner 作为主方案**，因为那已经偏离 MAGPIE。

---

# 11. Load-balancing Loss

必须加入。

否则 Router 很容易：

```text
E1 selected 70%
E2 selected 20%
其余 6 experts 很少使用
```

形成 Expert Collapse。

第一版实现 MAGPIE / Switch-style routing balance loss。

代码必须统计：

```text
expert_selection_count
expert_selection_ratio
mean_router_probability
```

总 Loss：

```math
L =
L_cls
+ λ_balance L_balance
```

如果当前代码已有 entropy regularization，可做可选消融，但主版本先保持和 MAGPIE 核心一致，不增加过多路由 loss。

---

# 12. STEMI / NSTEMI / UA 输出

## 12.1 全 19,955 ECG 三标签模式

输出：

```text
3 logits
→ sigmoid
```

对应：

```text
p_STEMI
p_NSTEMI
p_UA
```

Loss 推荐：

```text
Weighted BCE
```

或当前最佳：

```text
Asymmetric Loss
```

禁止因为新模型而同时改变旧 baseline 的 loss，否则无法公平比较。

---

## 12.2 ACS-only 三分类模式

如果只用：

```text
8,984 ACS ECG
```

则：

```text
3 logits
→ softmax
```

Loss：

```text
class-weighted CE
```

两种模式都保留。

---

# 13. 先做标签审计

数据论文报告：

```text
STEMI  = 1513
NSTEMI = 1274
UA     = 6197
ACS    = 8984
```

且：

```text
1513 + 1274 + 6197 = 8984
```

因此三类高度可能互斥。

训练前必须保存：

```text
label_audit.json
label_cooccurrence.csv
```

统计：

```text
STEMI & NSTEMI
STEMI & UA
NSTEMI & UA
triple positive
all-zero
```

不能假设。

---

# 14. 分类阈值必须优化

全队列三 sigmoid 模式禁止只使用：

```text
0.5
```

作为最终阈值。

每个 fold：

```text
Train
↓
Validation
↓
learn t_STEMI / t_NSTEMI / t_UA
↓
freeze thresholds
↓
Test
```

必须保证：

> **Test fold 不参与阈值选择。**

支持：

```text
fixed_0.5
youden_j
max_f1
balanced_accuracy
```

---

# 15. NSTEMI vs UA 是重点子任务

虽然总体预测：

```text
STEMI / NSTEMI / UA
```

但必须单独重点报告：

```text
NSTEMI vs UA
```

因为两者都不要求持续性 ST elevation，ECG 区分通常更加困难，而临床定义的关键差异又涉及 myocardial necrosis。

必须在真实标签属于：

```text
NSTEMI or UA
```

的子集单独报告：

```text
AUROC
AUPRC
F1
Balanced Accuracy
Sensitivity_NSTEMI
Sensitivity_UA
Confusion Matrix
```

---

# 16. NSTEMI / UA 阈值联合优化

在验证集上支持：

```math
(t_N*, t_U*) =
argmax Metric_NSTEMI_vs_UA(t_N, t_U)
```

推荐目标：

```text
Macro F1
```

或：

```text
Balanced Accuracy
```

搜索：

```text
0.05 ~ 0.95
```

Test fold 只使用已经冻结的阈值。

---

# 17. 重点创新实验：KAN Experts 替换 MLP Experts

这是本版最重要的新实验。

## 17.1 原则

除了 Expert 类型，其他所有配置必须完全一致：

```text
same Xception backbone
same Router
same num_experts = 8
same Top-2
same routing loss
same folds
same optimizer
same scheduler
same augmentation
same classifier head
same thresholds
```

唯一变化：

```text
MLP/FFN Experts
        ↓
KAN Experts
```

这样才能回答：

> KAN 作为 Sparse MoE Expert 是否优于传统 FFN Expert？

---

# 18. KAN-MoE Variant

模型：

```text
                         ECG
                          │
                     Xception
                          │
                    ECG Tokens
                          │
                       Router
                          │
                   scores for 8
                          │
                       Top-2
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
      Selected KAN Expert        Selected KAN Expert
            │                           │
            └──── weighted fusion ──────┘
                          │
                       pooling
                          │
                    same linear head
                          │
                STEMI / NSTEMI / UA
```

---

# 19. KAN Expert 的实现原则

KAN 不直接吃：

```text
12 × 5000 raw ECG
```

而是处理已经由 Xception 提取的低维 token：

```text
h_t ∈ R^D
```

实现接口必须与 MLP Expert 一致：

```python
[B, L, D] -> [B, L, D]
```

保证 Router 和后续融合代码完全复用。

第一版 KAN 不要过大。

要求：

```text
KAN total params
≈ MLP total params
```

或者至少报告：

```text
total parameters
active parameters
FLOPs
latency
GPU memory
```

防止 KAN 只是因为参数更多而涨点。

---

# 20. MLP-MoE vs KAN-MoE 的公平实验

必须比较：

| Model | Experts | Router | Top-K | 其他 |
|---|---|---|---|---|
| M0 | 无 MoE | 无 | - | Xception baseline |
| M1 | 8 MLP Experts | 1 Router | Top-2 | MAGPIE-style |
| M2 | 8 KAN Experts | 1 Router | Top-2 | 其余同 M1 |

核心：

```text
M1 vs M0
→ Sparse MoE 是否有效

M2 vs M1
→ KAN Expert 是否比 MLP Expert 更好
```

---

# 21. 如果 KAN 更好，创新点怎么写

只有满足以下条件：

1. 五折大多数 fold 均提升；
2. Macro AUPRC / Macro F1 至少有稳定提升；
3. NSTEMI vs UA 指标也不下降，最好提高；
4. 参数量 / 计算量差异已公平报告；
5. 不是仅靠 threshold 调整产生提升；

才可以写：

> We replace conventional FFN experts in a MAGPIE-style sparse MoE with KAN experts to enhance nonlinear expert transformations for ECG representation learning.

更具体：

> A KAN-based sparse expert layer is introduced to model nonlinear transformations of ECG temporal tokens while retaining Top-2 conditional computation.

不要提前写：

```text
KAN 一定优于 MLP
```

---

# 22. 必做消融实验

## 22.1 MAGPIE Sparse MoE 主框架消融

| ID | 配置 | 目的 |
|---|---|---|
| A | Xception baseline | 基准 |
| B | Xception + Dense MLP block | 增加参数是否有效 |
| C | Xception + 8 Experts, all-active | Dense MoE |
| D | Xception + 8 Experts, Top-1 | Sparse K |
| E | **Xception + 8 MLP Experts, Top-2** | MAGPIE-style 主模型 |
| F | Xception + 8 Experts, Top-3 | Top-K 消融 |
| G | E - load balance | 证明负载均衡作用 |

---

## 22.2 Expert 数量消融

至少：

```text
4 Experts
8 Experts
```

资源允许再：

```text
12 Experts
```

Top-K 固定 2。

目的：

> 验证 8 Experts 是否适合 ECG，而不是因为 MAGPIE 用 8 就直接认为最优。

---

## 22.3 KAN 消融

| ID | Expert 类型 | 其他 |
|---|---|---|
| K0 | MLP / FFN | Top-2 |
| K1 | **KAN** | Top-2 |

额外可选：

```text
MLP parameter-matched
KAN parameter-matched
```

---

## 22.4 Routing Level 消融

比较：

```text
global-vector routing
vs
temporal-token routing
```

如果 token-level routing 明显更好，则说明：

> 不同 ECG 时间区域需要不同 Expert specialization。

---

## 22.5 Threshold 消融

```text
fixed 0.5
per-class max F1
per-class Youden J
per-class balanced accuracy
NSTEMI-UA joint tuning
```

注意：

> threshold tuning 只能改善决策点，不代表 representation 变好。

所以主要模型优劣仍以：

```text
AUROC
AUPRC
```

等 threshold-free 指标为重要依据。

---

# 23. Router / Expert 可解释性

每个 fold 保存：

```text
router_scores.csv
expert_selection.csv
expert_usage.csv
```

必须分析：

## 23.1 Expert usage

```text
P(E_i selected)
```

## 23.2 按类别统计

```text
P(E_i selected | STEMI)
P(E_i selected | NSTEMI)
P(E_i selected | UA)
```

虽然 Router 不以类别为条件，但可以后验分析不同类别是否自然形成不同 Expert 偏好。

这非常重要，因为它能够验证：

> MAGPIE-style implicit specialization 是否也会在 ACS ECG 中出现。

---

# 24. NSTEMI vs UA 的 Expert 专门化分析

重点比较：

```text
Expert selection distribution for NSTEMI
vs
Expert selection distribution for UA
```

建议：

```text
Jensen-Shannon divergence
```

或：

```text
cosine distance
```

并报告五折 mean ± std。

如果两类形成稳定不同的 expert usage pattern，这是很有论文价值的结果。

---

# 25. 训练 Loss

主模型：

```math
L =
L_cls
+
λ_bal L_balance
```

全队列三标签若审计确认互斥，可加入：

```math
L =
L_cls
+
λ_bal L_balance
+
λ_excl L_excl
```

但：

```text
L_excl
```

必须作为单独消融。

---

# 26. Patient-level 5-fold Cross Validation

必须按 Patient ID 分。

不能：

```text
ECG record random KFold
```

因为：

```text
19,955 ECG
18,909 patients
```

说明存在同患者多 ECG。

严格保证：

```text
train_patient ∩ val_patient = ∅
train_patient ∩ test_patient = ∅
```

推荐：

```text
StratifiedGroupKFold
```

---

# 27. 评价指标

## 主标签

每类：

```text
AUROC
AUPRC
F1
Sensitivity
Specificity
Precision / PPV
NPV
```

整体：

```text
Macro AUROC
Macro AUPRC
Macro F1
Micro F1
```

主指标：

```text
Macro AUPRC
Macro F1
```

---

## NSTEMI vs UA

单独：

```text
AUROC
AUPRC
F1
Balanced Accuracy
Sensitivity
Specificity
Confusion Matrix
```

---

# 28. KAN 是否成为创新点的 Go / No-Go 标准

## Go

如果：

```text
KAN-MoE > MLP-MoE
```

满足：

- 大多数 fold 同方向；
- Macro AUPRC 提升；
- Macro F1 提升或不下降；
- NSTEMI vs UA 至少一项关键指标明显提升；
- active parameter / FLOPs / latency 可接受；

则 KAN Expert 可以进入最终论文创新点。

## No-Go

如果：

- 只在 1 个 fold 提升；
- 只靠 threshold 调整提高；
- AUPRC 不变但参数量大幅增加；
- NSTEMI vs UA 下降；
- 训练不稳定；

则保留 MAGPIE-style MLP Sparse MoE，删除 KAN。

---

# 29. Codex 修改旧模型的明确要求

1. 扫描当前旧 MoE 与 Xception 实现。
2. 不删除任何旧模型。
3. 保留原数据 preprocessing、augmentation、optimizer、scheduler 和 best training strategy。
4. 新增 `acs_magpie_moe` 模型。
5. Xception 去掉原分类头，输出 temporal feature map。
6. 添加 projection，得到 `[B,L,D]` tokens。
7. 实现一个 Router：
   ```text
   D → 8 logits
   ```
8. 实现 8 个独立 MLP/FFN Experts。
9. 实现 Noisy Top-2 train / deterministic Top-2 eval。
10. 实现 selected-expert weighted aggregation。
11. 实现 residual connection。
12. 加入 load-balance loss。
13. pooling 后使用与 baseline 相同的 3-output classification head。
14. 实现 KAN Expert，与 MLP Expert 保持完全相同输入/输出接口。
15. 通过 config 切换：
   ```text
   expert_type = mlp | kan
   ```
16. 支持：
   ```text
   num_experts
   top_k
   routing_level
   ```
17. 做 patient-level 5-fold。
18. 做 validation-only threshold optimization。
19. 单独输出 NSTEMI vs UA 指标。
20. 保存 Router / Expert usage。
21. 完成本文所有消融实验。
22. 最终输出 MLP-MoE vs KAN-MoE 的公平比较。

---

# 30. 推荐配置

```yaml
model:
  name: acs_magpie_moe

  backbone:
    name: xception
    pretrained: false
    output_mode: temporal_feature_map

  moe:
    num_experts: 8
    top_k: 2
    router_type: noisy_topk
    routing_level: token
    expert_type: mlp
    model_dim: 256
    ff_hidden_multiplier: 4
    load_balance_weight: 0.01

  classifier:
    type: linear
    num_outputs: 3

task:
  mode: multilabel3_full_cohort
  labels:
    - STEMI
    - NSTEMI
    - UA

thresholding:
  fixed_baseline: 0.5
  tune_on_validation: true
  strategies:
    - max_f1
    - youden_j
    - balanced_accuracy
  nstemi_ua_joint_tuning: true
```

KAN 对照：

```yaml
model:
  moe:
    expert_type: kan
```

除该项外，其他配置尽量完全不变。

---

# 31. 自动测试

必须增加：

### Shape

```text
Input  [B,12,5000]
Tokens [B,L,D]
Output [B,3]
```

### Router

```text
每个 token exactly Top-2
```

### Expert usage

训练一个小 batch 后：

```text
selection_count.sum() == B * L * top_k
```

### MLP/KAN interface

```text
MLP Expert: [B,L,D] → [B,L,D]
KAN Expert: [B,L,D] → [B,L,D]
```

### Patient leakage

所有 fold 无患者重叠。

### Threshold leakage

Test predictions 不得参与 threshold fitting。

---

# 32. 实验产物

每 fold：

```text
metrics.json
predictions.csv
thresholds.json
router_scores.csv
expert_selection.csv
expert_usage.csv
```

总体：

```text
cv_summary.csv
moe_ablation.csv
topk_ablation.csv
expert_count_ablation.csv
kan_vs_mlp.csv
nstemi_vs_ua_summary.csv
threshold_summary.csv
expert_usage_by_class.csv
```

图：

```text
expert_usage_heatmap.png
nstemi_vs_ua_confusion_matrix.png
router_distribution_by_class.png
kan_vs_mlp_fold_comparison.png
```

---

# 33. 论文依据

## R1. 数据集

Du X, Liu Y, Wang L, et al.  
*A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome prediction containing 19,955 ECGs.*  
**Scientific Data**, 2026.  
DOI: `10.1038/s41597-026-07278-0`

用途：

- 数据集；
- STEMI / NSTEMI / UA 标签；
- patient-level CV 必要性；
- 类别不平衡。

---

## R2. MAGPIE —— 本研究最主要的方法依据

Xie M, Xiao Q, Wu H, et al.  
*Masked autoencoding, generalizable pretraining, and integrated experts for enhanced glioma segmentation.*  
**npj Digital Medicine**, 2026;9:163.  
DOI: `10.1038/s41746-026-02347-5`

核心：

```text
Sparse MoE
8 FFN Experts
Top-2 routing
Load balancing
Top-K ablation
Expert specialization analysis
```

本研究直接迁移：

```text
8 Experts
One Router
Top-2
Load Balance
Expert usage analysis
```

---

## R3. MMoE —— 多任务专家学习背景

Ma J, Zhao Z, Yi X, et al.  
*Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts.*  
**KDD 2018**.  
DOI: `10.1145/3219819.3220007`

用途：

> 说明 MoE 可以缓解共享表示在相关任务之间的 interference。

本版主模型不直接使用 multi-gate，只作为理论背景。

---

## R4. CardX / ExChanGeAI preprint —— ECG Sparse MoE 补充依据

*ExChanGeAI: An End-to-End Platform and Efficient Foundation Model for Electrocardiogram Analysis and Fine-tuning.*  
arXiv:`2503.13570`.

CardX 预印本展示了：

```text
ECG
+ heterogeneous architectures
+ sparse / Top-K routing
```

用途：

> 证明 Sparse architecture routing 可迁移到 ECG。

注意：不是正式高 IF 期刊主依据。

---

## R5. ECG-MoE —— ECG Expert specialization 补充依据

*ECG-MoE: Mixture-of-Expert Electrocardiogram Foundation Model.*  
NeurIPS TS4H Workshop / arXiv:`2603.04589`.

用途：

> ECG Expert 可以通过训练形成 morphology / rhythm 等 specialization。

---

## R6. xGNN4MI —— 12 导联空间关系背景

*xGNN4MI: explainability of graph neural networks in 12-lead electrocardiography for cardiovascular disease classification.*  
**npj Digital Medicine**, 2026.  
DOI: `10.1038/s41746-026-02367-1`

本版不使用 GAT Expert，但该论文支持：

> 12 导联 ECG 的空间关系具有独立判别信息。

可以作为后续扩展背景。

---

## R7. Wavelet ECG / MI —— 时频背景

包括：

- CWT + Multi-branch Transformer, **Heliyon**, 2024；
- DWT + lightweight CNN for MI, **BMC Medical Imaging**, 2024；
- CWT / STFT for STEMI culprit-lesion prediction。

本版不把 Wavelet 作为 Expert。

用途：

> 说明 ECG 时频信息是后续可扩展方向，而不是当前主框架的必要模块。

---

## R8. FuseMoE —— 临床 Sparse MoE 背景

*FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion.*  
**NeurIPS 2024**.

其临床实验包含 MIMIC-IV-ECG。

用途：

> 支持 Sparse MoE 可用于临床数据和 ECG representation。

---

## R9. MoViE —— Sparse MoE + MI

*MoViE: A Mixture-Of-Experts Multimodal Fusion Model for Cardiovascular Disease Detection.*  
**IEEE SMC 2025**.

用途：

> Sparse Experts 已被用于 MI-related detection，说明本研究不是脱离 ECG / ACS 场景的纯算法迁移。

---

## R10. ECG 多标签互斥关系

*Multi-expert ensemble ECG diagnostic algorithm using mutually exclusive–symbiotic correlation between 254 hierarchical multiple labels.*  
**npj Cardiovascular Health**, 2024.  
DOI: `10.1038/s44325-024-00010-0`

用途：

> 支持 ECG 标签之间的互斥 / 共生关系建模和标签审计。

---

## R11. KAN ECG 依据

*Feature extraction and intelligent diagnosis of ECG signals based on KANs and xLSTM.*  
**Biosensors and Bioelectronics**, 2026;292:118105.  
DOI: `10.1016/j.bios.2025.118105`

以及：

*MAK-Net: A Multi-Scale Attentive Kolmogorov–Arnold Network with BiGRU for Imbalanced ECG Arrhythmia Classification.*  
**Sensors**, 2025.

用途：

> KAN 已经具有 ECG 建模依据。

本研究进一步验证：

```text
KAN 是否能替代 Sparse MoE 中的传统 FFN Expert
```

这是一个需要实验验证的新问题。

---

# 34. 论文创新边界

不要写：

```text
首次将 Sparse MoE 用于 ECG
首次将 KAN 用于 ECG
```

更稳妥的创新表述：

> We transfer a MAGPIE-style Top-2 sparse mixture-of-experts architecture to large-scale 12-lead ACS ECG classification and investigate whether replacing conventional FFN experts with KAN experts improves nonlinear ECG token specialization, with particular emphasis on differentiating NSTEMI from UA.

中文：

> 本研究首先将 MAGPIE 式 Top-2 Sparse MoE 迁移至大规模 12 导联 ACS ECG 分类，在固定 Xception 特征提取器、Router、专家数量及训练策略的条件下，进一步探索以 KAN Expert 替代传统 FFN Expert 是否能够增强 ECG token 的非线性专家专门化能力，并重点评估其对 NSTEMI 与 UA 精细区分的作用。

---

# 35. 最终研究路线

```text
Step 1
Xception baseline

Step 2
Xception + MAGPIE-style 8 MLP Experts + Top-2
→ 验证 Sparse MoE 是否有效

Step 3
Top-1 / Top-2 / Top-3
→ 验证 K

Step 4
4 / 8 / 12 Experts
→ 验证专家数量

Step 5
MLP Expert → KAN Expert
→ 唯一变量对照

Step 6
Validation-only threshold optimization

Step 7
重点分析 NSTEMI vs UA

Step 8
Expert routing / specialization analysis
```

如果：

```text
KAN-MoE > MLP-MoE
```

且结果跨五折稳定，则最终模型为：

```text
Xception
→ MAGPIE-style Sparse Top-2
→ 8 KAN Experts
→ STEMI / NSTEMI / UA
```

如果 KAN 无稳定优势，则最终模型保留：

```text
Xception
→ MAGPIE-style Sparse Top-2
→ 8 MLP/FFN Experts
→ STEMI / NSTEMI / UA
```

这保证论文结论由实验决定，而不是为了保留创新点强行堆模块。
