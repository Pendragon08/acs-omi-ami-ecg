# ACS 多标签分类：Xception + Heterogeneous MMoE + KAN 可行性实验报告（历史 AMI/OMI 方案）

> **当前任务定义（2026-08-30）**：本文原先的 AMI/OMI 双独立任务以及
> STEMI/NSTEMI/UA 独立 sigmoid 试验均为历史记录。正式主线现在是 ACS
> `OTHER/STEMI/NSTEMI/UA` 互斥单标签四分类 + 独立 OMI/NOMI 二分类联合任务。
> 当前标签、损失和评价以 `CORRECTED_TASK_FORMULATION.md` 为准。

> **方案修订（2026-08-25）**：依据 Kim 等人的 WavelNet 论文及其官方 GitHub 实现，Frequency Expert 已由“MVMD + 外部缓存 + CNN”改为端到端 **WavelNet WaveletConv**。本报告中旧的 MVMD 流程、MVMD 参数搜索、MVMD 缓存和 MVMD 消融均以本修订为准，禁止在网络外部执行 MVMD/VMD 分解。

论文与代码：

- 论文：<https://doi.org/10.1016/j.cmpb.2023.107375>
- 官方实现：<https://github.com/NamhoKim-Kor/WavelNet>（WaveletConv 层，MIT License）
- 本项目适配：对 12 个导联共享同一个 16-filter Symlet-4 WaveletConv，再接论文风格的 Conv/MaxPool 编码器；输入仍沿用项目已冻结的 P3 数据路径，频率分析在模型内部端到端学习。

当前 ACS 预处理数据核对结果：训练记录 `17,960` 条、唯一患者 `17,018` 人，数组形状 `[17,960,1000,12]`，采样率 `100 Hz`；AMI 阳性 `2,679`（14.916%），OMI 阳性 `1,151`（6.409%）。AMI/OMI 患者级联合状态计数为 `[14,378, 1,501, 0, 1,139]`（按 `[0,0]、[1,0]、[0,1]、[1,1]` 编码），其中 `[0,1]` 是数据中不存在的空状态。因此五折分层只对非空状态检查样本数，不能把空状态误报为划分失败；所有实际存在状态均满足五折要求。

实验汇总报告：[`reports/acs_hmoe_kan_feasibility.md`](reports/acs_hmoe_kan_feasibility.md)。

当前执行记录（2026-08-25，远程 RTX A6000）：GPU0 的 STEMI/NSTEMI/UA 三标签最终方案已完成 `XceptionTime + P3 + BCE, seed=2026` 单次训练；验证集 macro-AUROC=`0.78965`、macro-AUPRC=`0.47655`、macro-F1=`0.51376`，详细指标和 checkpoint 保存在服务器 `results/stemi_nstemi_ua_final_gpu0/`。已用验证集阈值固定生成外部测试集 `1,995` 条记录，并将合并 CSV 提交至 ACS 平台盲评；平台返回 STEMI/NSTEMI/UA 的 Sensitivity 分别为 `0.7099/0.3235/0.8886`，Specificity=`0.9624/0.9414/0.3382`，F1=`0.6647/0.3045/0.5619`。平台页面和提交文件记录在 `reports/acs_hmoe_kan_feasibility.md`；这些分数是平台盲评反馈，不是本地可复算的有标签测试统计。GPU1 正在按 E0→E4 顺序运行本报告的 AMI/OMI HMoE 可行性实验，Frequency Expert 使用 WavelNet，未使用 MVMD。

> **用途**：本报告直接交给 Codex 使用。请先扫描当前项目代码、数据加载流程、现有最佳 XceptionNet 配置和训练脚本，再在**不破坏现有最佳基线**的前提下实现并验证本方案。
>
> **历史任务**：基于 12 导联 ECG 联合预测两个二分类标签：`AMI` 与 `OMI`（已归档）。
>
> **历史核心目标**：验证“异构专家 + AMI/OMI 双 Router + KAN 分类头”是否比旧 XceptionNet 多标签基线稳定提升；该结论不替代当前 ACS/OMI 联合任务。

---

## 1. 研究问题

本实验不是单纯测试“堆叠更多网络是否涨点”，而是验证以下假设：

1. **单一 XceptionNet 表征可能不足以同时覆盖 12 导联 ECG 的不同信息来源**，包括原始形态、跨导联空间关系和非平稳频率/模态信息。
2. **AMI 与 OMI 高度相关但并不完全等价**，强制二者共享完全相同的高级表征可能导致任务干扰或负迁移。
3. 不同患者的 ECG 表型存在异质性，因此同一任务对不同专家的依赖也可能不同。
4. 异构 Expert Bank + Task-specific Router 可以让 AMI 与 OMI 从共享专家池中自适应选择不同特征组合。
5. KAN 可用于最终任务分类头，验证其是否比传统 MLP/Linear Head 更适合建模融合特征与 AMI/OMI 标签之间的复杂非线性映射。

最终要回答：

> **Xception + Heterogeneous MMoE + KAN 是否能在患者级五折交叉验证中稳定优于 Xception 多标签基线？如果能，性能提升主要来自哪个模块？Router 是否真的学到了 AMI 与 OMI 不同的专家偏好？**

---

# 2. 历史 AMI/OMI 任务定义（已归档）

> 本节记录早期 AMI/OMI 双二分类设计，仅用于复现实验历史。正式实验请使用
> 下方“当前 ACS/OMI 联合任务”定义。

## 2.1 输入

12 导联 ECG：

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

输入张量按当前项目数据格式处理，典型形式：

```text
[B, 12, T]
```

如果原始记录为 10 秒、500 Hz，则 `T=5000`；不要硬编码，必须从现有数据加载器读取实际长度。

## 2.2 输出（历史）

两个独立二分类标签：

```text
[AMI, OMI]
```

模型输出两个 logits / probabilities：

```text
P(AMI)
P(OMI)
```

最终层使用两个独立 Sigmoid，**不要使用 2 类 Softmax**。

## 2.3 标签约束（历史）

- 必须直接使用当前数据集 metadata 中定义的 `AMI` 与 `OMI` 标签。
- 不要自行通过 STEMI/NSTEMI/UA 推导 AMI 标签，除非当前项目本身就是按数据集官方定义生成标签。
- 不要根据 OMI/AMI 的医学关系人为修正标签。
- 先统计真实标签联合分布：

```text
AMI=0, OMI=0
AMI=1, OMI=0
AMI=0, OMI=1
AMI=1, OMI=1
```

并在实验报告中输出患者数和 ECG 数。

## 2.4 当前 ACS/OMI 联合任务

当前模型输出两个头：

```text
ACS head: OTHER / STEMI / NSTEMI / UA  （4-way softmax/argmax）
OMI head: OMI / NOMI                    （独立 sigmoid）
```

STEMI、NSTEMI、UA 互斥；源 `[0,0,0]` 映射为 `OTHER` 并保留，`OTHER + OMI`
是有效组合。ACS 使用交叉熵，OMI 使用稳定的 `BCEWithLogitsLoss`，只有 OMI
阈值可在验证 OOF 上确定。当前代码和结果以
`CORRECTED_TASK_FORMULATION.md` 为准。

---

# 3. 总体模型方案

建议模型名称暂定：

**ECG-HMoE-KAN**

全称：

**Heterogeneous Multi-Gate Mixture-of-Experts with KAN Classifiers for Joint AMI and OMI Detection**

整体框架：

```text
                           12-lead ECG
                                │
        ┌───────────────────────┼────────────────────────┐
        │                       │                        │
        ▼                       ▼                        ▼
 Raw Morphology Branch     Lead-Spatial Branch      Frequency Branch
        │                       │                        │
        ▼                       ▼                        ▼
   1D XceptionNet        Lead-wise Encoder          WavelNet WaveletConv
         │                       │                        │
         │                       ▼               shared Symlet-4 filters
         │                    GAT/GNN             per lead, 16 kernels
         │                       │                        │
         │                       │                Conv/MaxPool encoder
        │                       │                            │
        ▼                       ▼                            ▼
 Morphology Feature      Lead-Spatial Feature        Frequency Feature
      z_morph                  z_lead                     z_freq
        │                       │                            │
        └───────────────────────┼────────────────────────────┘
                                ▼
                      Unified Projection
                          same dim D
                                │
                                ▼
                         Expert Bank
               {E_morph, E_lead, E_freq}
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
               AMI Router                OMI Router
                   │                         │
            αA_m, αA_l, αA_f        αO_m, αO_l, αO_f
                   │                         │
                   ▼                         ▼
            Weighted Fusion           Weighted Fusion
                   │                         │
                 z_AMI                     z_OMI
                   │                         │
                   ▼                         ▼
               KAN_AMI                   KAN_OMI
                   │                         │
                Sigmoid                   Sigmoid
                   │                         │
                   ▼                         ▼
                P(AMI)                    P(OMI)
```

---

# 4. 三个专家的具体设计

## 4.1 Expert 1：Morphology Expert — XceptionNet

### 目的

学习原始 12 导联 ECG 的层级形态特征，包括但不限于：

- QRS morphology
- ST-T morphology
- 局部缺血相关波形模式
- 不同导联内部的局部时间特征

注意：代码和论文中可以写“designed to capture morphological patterns”，但**不要在训练前声称模型已经学会某种具体临床模式**。

### 实现原则

优先复用当前项目中已经验证最好的 XceptionNet：

```text
Raw ECG
  ↓
Current Best XceptionNet
  ↓
Remove Original Classification Head
  ↓
Global Pooling / Existing Feature Output
  ↓
z_morph
```

要求：

- 不改变当前最佳 Xception 主干结构，除非为暴露 feature embedding 必须做最小修改。
- 保留当前最佳归一化、激活、dropout 和 residual 设置。
- 输出高层特征，不直接分类。

---

## 4.2 Expert 2：Lead-Spatial Expert — GAT/GNN

### 目的

显式学习 12 导联之间的空间/相关关系，而不是把 12 导联完全视为独立通道。

### 推荐结构

```text
12-lead ECG
    ↓
Shared lead encoder
每个导联独立/共享参数编码
    ↓
12 lead embeddings
    ↓
GAT / lightweight GNN
    ↓
Graph pooling
    ↓
z_lead
```

### 推荐实现

第一版优先使用轻量 GAT：

```text
Lead Encoder: lightweight 1D CNN or shallow separable CNN
Node Count: 12
Node Feature: D_lead
GAT Layers: 1~2 layers
Graph Pooling: mean / attention pooling
```

图结构有两种实现方式：

**优先方案 A：Fully-connected learned attention graph**

- 12 个导联全部可互相 attention。
- 不人为写死临床 lead adjacency。
- 让 GAT 学习导联相关性。

**备选方案 B：Clinical prior graph**

只有在当前项目已有明确导联图定义或已有代码时再使用。

第一轮实验优先 A，减少人为规则。

---

## 4.3 Expert 3：Frequency Expert — WavelNet WaveletConv + Lightweight CNN

### 目的

学习非平稳 ECG 中不同模态/频率成分提供的判别信息。

### 推荐流程（端到端，不做外部 MVMD/VMD）

```text
12-lead ECG
    ↓
shared WavelNet WaveletConv (Symlet-4, 16 filters, kernel=103)
    ↓
per-lead wavelet activations [12 × 16 × T]
    ↓
lead mixing + Conv/MaxPool encoder
    ↓
z_freq
```

### 为什么采用 WavelNet

论文将 WaveletConv 放在网络第一层，用不同尺度的母小波核对原始时域信号做自适应时频分析；尺度参数随训练更新，因此不需要 MVMD/VMD 的外部分解、模态排序或缓存。Symlet-4 在原论文中表现最好，本实验固定使用 Symlet-4 作为第一版频率专家，避免把母小波选择与 HMoE 结构同时搜索。

### ACS 适配与固定配置

- 采样率按项目数据为 `fs=100 Hz`，输入长度为 `T=1000`。
- WaveletConv 使用官方 GitHub 实现，16 个共享滤波器，奇数 kernel length `103`；对 12 个导联逐导联共享应用。
- 本 ACS 适配固定 `a_min=0.2`，其余尺度由官方 WaveletConv 的可学习参数 `a` 更新；Symlet-4 母小波由 PyWavelets 的 canonical `sym4` 原型归一化到零均值/单位范数后重采样到 103 点。
- 通过轻量 `1×1` lead mixing 合并 12×16 个激活，再使用论文风格的 `Conv1d(16/32/64, kernel=5) + ReLU + MaxPool(stride=4)` 编码到 `z_freq`。
- 不新增 MVMD/VMD 数组，不使用 `MVMD_K4_train.npy`，不把任何测试标签用于小波参数设置。
- 当前项目的 P3 数据链路（P2 的 0.5–40 Hz band-pass + train-fold robust scaling + train-only augmentation）保持不变；因此本实验验证的是“在既有 ACS 预处理下加入 WavelNet 频率专家”的可行性。论文原始 WavelNet 使用 raw ECG、无 BN；这一差异在报告中单独记录，不冒充完全复现原论文。

### Frequency Encoder

不使用完整大 Xception 重复多个模态，也不把频率分解放在网络外部。直接复用官方 WavelNet `WaveletConv`，随后使用共享的轻量 Conv/MaxPool 编码器；所有 WaveletConv scale 参数随分类损失端到端更新。

---

# 5. Feature Projection

三个专家输出维度可能不同：

```text
z_morph: D1
z_lead:  D2
z_freq:  D3
```

必须分别投影到统一维度：

```text
p_morph = P_morph(z_morph)
p_lead  = P_lead(z_lead)
p_freq  = P_freq(z_freq)
```

统一：

```text
D = 256
```

如果当前 Xception 最佳 embedding 维度已较小，可根据现有结构选择 `D=128/256/512`，但第一轮固定一个值，不做大规模搜索。

Projection 推荐：

```text
Linear
→ LayerNorm
→ GELU
```

---

# 6. AMI / OMI 双 Router

这是本方案核心。

## 6.1 输入

将三个投影后的 expert feature 拼接：

```text
z_all = concat(p_morph, p_lead, p_freq)
```

## 6.2 AMI Router

```text
g_AMI = Softmax(Router_AMI(z_all))
```

输出：

```text
[B, 3]
```

例如：

```text
[morphology, lead-spatial, frequency]
[0.50, 0.31, 0.19]
```

AMI 融合：

```text
z_AMI =
  g_AMI[:,0] * p_morph
+ g_AMI[:,1] * p_lead
+ g_AMI[:,2] * p_freq
```

## 6.3 OMI Router

完全独立参数：

```text
g_OMI = Softmax(Router_OMI(z_all))
```

OMI 融合：

```text
z_OMI =
  g_OMI[:,0] * p_morph
+ g_OMI[:,1] * p_lead
+ g_OMI[:,2] * p_freq
```

核心假设：

```text
G_AMI(X) != G_OMI(X)
```

第一版使用 **Dense Soft Routing**。

不要先使用 Top-K Sparse Routing。

---

# 7. KAN 分类器

## 7.1 设计

Router 输出任务特异特征后，分别进入两个独立 KAN：

```text
z_AMI → KAN_AMI → logit_AMI
z_OMI → KAN_OMI → logit_OMI
```

最终：

```text
p_AMI = sigmoid(logit_AMI)
p_OMI = sigmoid(logit_OMI)
```

## 7.2 要求

- `KAN_AMI` 与 `KAN_OMI` 参数完全独立。
- KAN 不直接处理 `[12, T]` 原始 ECG。
- KAN 只处理已经压缩后的任务特异 feature，例如 256-D。
- 优先使用项目中已有 KAN 实现。
- 如果项目没有 KAN：
  1. 先检查当前环境可兼容的成熟实现；
  2. 固定并记录依赖版本；
  3. 不要偷偷用 MLP 替代 KAN；
  4. 如果 KAN 依赖无法稳定运行，则将“MMoE + MLP Head”作为完整对照，并在报告中明确 KAN 未完成的原因。

---

# 8. Loss

第一轮尽量简单，不要堆很多新 loss。

分类损失：

```text
L_cls = λ_A * L_AMI + λ_O * L_OMI
```

`L_AMI` 和 `L_OMI`：

- 优先沿用当前项目已有最佳分类 loss。
- 如果现有 loss 为普通 BCE 且类别明显不平衡，可增加 Weighted BCE / Asymmetric Loss 实验，但不要和 MoE 主实验同时大规模调参。

增加轻量 Router balance loss：

```text
L_total = L_cls + λ_balance * L_balance
```

目的：防止专家塌缩。

要求：

- 记录每折每个 expert 的平均权重和使用率。
- `λ_balance` 只在训练 fold / validation 内选择。
- 不要求每个 Expert 恰好 1/3 使用率，只要求避免长期完全塌缩。

第一版不要加入：

- Top-K sparse loss
- label consistency loss
- clinical-guided router
- uncertainty loss
- auxiliary vessel loss
- contrastive loss

这些全部留到主方案验证成功以后。

---

# 9. 数据划分：必须患者级五折

这是强制要求。

## 9.1 禁止普通 ECG-level StratifiedKFold

如果同一患者有多份 ECG，不能让其同时出现在训练和测试中。

必须使用：

```text
Patient-level 5-fold cross-validation
```

同一患者所有 ECG 必须进入同一 fold。

## 9.2 多标签分布保持

划分时尽可能保持以下四种联合标签比例：

```text
AMI0 OMI0
AMI1 OMI0
AMI0 OMI1
AMI1 OMI1
```

在 5 folds 中接近。

如果项目已有固定患者级 split，优先复用并保存 split manifest。

必须将最终 fold 划分保存到：

```text
artifacts/splits/acs_ami_omi_5fold.csv
```

字段至少包括：

```text
patient_id
ecg_id
fold
ami_label
omi_label
```

后续所有实验必须使用**完全相同的 fold**。

---

# 10. 训练与验证规则

1. 所有模型使用相同患者级五折。
2. 所有模型尽可能沿用当前最佳：
   - optimizer
   - learning rate
   - scheduler
   - batch size
   - augmentation
   - epoch / early stopping
   - preprocessing
3. 只有新增模块允许增加单独 learning rate group；若使用必须记录。
4. 每个 fold：
   - train
   - validation
   - held-out test
5. test fold 只在最终一次评估时使用。
6. F1/Sensitivity/Specificity 的 threshold 必须在 validation set 确定，禁止在 test set 调 threshold。
7. AUROC/AUPRC 使用连续概率直接计算。
8. 保存每折最佳 checkpoint。
9. 固定随机 seed，并记录到配置。
10. 第一轮每 fold 一个 seed 即可；如果最终模型明显有效，再追加 3 seeds 做稳定性验证。

---

# 11. 必须做的实验顺序

不要一次性只跑最终大模型。

## E0：Current Baseline — Xception + Two Heads

```text
ECG
 ↓
Xception
 ↓
Shared Feature
 ├→ AMI MLP/Linear Head
 └→ OMI MLP/Linear Head
```

目的：

- 复现当前最强 Xception multi-label baseline。
- 所有后续模型必须和 E0 使用相同 split、训练设置和数据增强。

---

## E1：Xception + KAN Heads

```text
ECG
 ↓
Xception
 ↓
Shared Feature
 ├→ KAN_AMI
 └→ KAN_OMI
```

目的：

> 单独验证 KAN 是否比现有 MLP/Linear Head 有增益。

如果 E1 不优于 E0，不要默认 KAN 是有效创新。

---

## E2：Homogeneous MMoE Baseline

```text
ECG
 ↓
Xception
 ↓
Shared Feature
 ↓
3 or 4 lightweight MLP Experts
 ↓
AMI Gate / OMI Gate
 ↓
MLP Heads
```

目的：

> 验证“AMI/OMI 双 Router MMoE”本身是否有效。

这里暂时不要 GAT、WavelNet、KAN。

---

## E3：Heterogeneous MMoE + MLP Heads

```text
Xception Expert
+
GAT Expert
+
WavelNet Frequency Expert
 ↓
AMI Router / OMI Router
 ↓
MLP Heads
```

目的：

> 验证异构 Expert Bank 是否优于同构 MLP-MMoE。

---

## E4：Final — Heterogeneous MMoE + KAN

```text
Xception Expert
+
GAT Expert
+
WavelNet Frequency Expert
 ↓
AMI Router / OMI Router
 ↓
KAN_AMI / KAN_OMI
 ↓
[AMI, OMI]
```

这才是最终完整方案。

---

# 12. 如果 E4 有效，继续做的消融实验

只在 E4 明显优于 E0 后继续。

## A1：Single Gate vs Dual Gate

```text
Shared Gate
vs
AMI Gate + OMI Gate
```

验证：

> AMI/OMI 是否真的需要任务特异 Router。

## A2：Uniform Fusion vs Learned Router

```text
1/3 + 1/3 + 1/3
vs
Learned Gate
```

验证：

> 提升是否来自动态 routing，而不是简单多分支集成。

## A3：Remove Frequency Expert

```text
Xception + GAT
```

验证 WavelNet Frequency Expert 对 AMI/OMI 的贡献。

## A4：Remove Lead-Spatial Expert

```text
Xception + WavelNet Frequency Expert
```

验证跨导联关系的贡献。

## A5：Remove Morphology Expert

```text
GAT + WavelNet Frequency Expert
```

验证原始 morphology 是否不可替代。

## A6：Remove Balance Loss

验证是否存在 Expert Collapse，以及 balance regularization 是否真正必要。

## A7：MLP Head vs KAN Head

保持所有 Expert/Router 完全一致，只替换最终 head。

这是 KAN 必须具备的核心消融。

---

# 13. 评价指标

## 13.1 每个标签分别报告

AMI：

```text
AUROC
AUPRC
F1
Sensitivity
Specificity
Precision / PPV
NPV
```

OMI：

```text
AUROC
AUPRC
F1
Sensitivity
Specificity
Precision / PPV
NPV
```

## 13.2 OMI 重点指标

OMI 类别通常更不平衡，因此不要只看 Accuracy。

重点排序：

```text
1. OMI AUPRC
2. OMI Sensitivity
3. OMI F1
4. OMI AUROC
5. Sensitivity @ fixed Specificity（如项目已有实现）
```

若实现 fixed-specificity 指标，阈值必须在 validation set 确定。

## 13.3 Multi-label overall

至少输出：

```text
Macro AUROC
Macro AUPRC
Macro F1
Micro F1
```

## 13.4 结果格式

每个实验输出：

```text
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
Mean ± Std
```

禁止只报告最佳 fold。

---

# 14. Router / Expert 分析：这是论文必做项

不能只看性能涨点。

## 14.1 Gate Weight Distribution

分别统计：

```text
AMI Router:
  Xception weight
  GAT weight
  WavelNet weight

OMI Router:
  Xception weight
  GAT weight
  WavelNet weight
```

分组分析：

```text
AMI+
AMI-
OMI+
OMI-
AMI+OMI+
AMI+OMI-
```

## 14.2 验证 AMI / OMI Gate 是否不同

至少计算：

```text
cosine distance
or
Jensen-Shannon divergence
```

比较：

```text
G_AMI(X)
vs
G_OMI(X)
```

如果两者几乎完全相同，则说明双 Gate 没有真正学出任务差异。

## 14.3 Expert Collapse

输出每个 expert：

```text
mean weight
median weight
std
fraction of samples where it is top-1 expert
```

如果某个专家长期接近 0 权重，标记为 under-utilized。

## 14.4 Expert Ablation

分别移除每个 Expert，比较 OMI/AMI 指标变化。

这是判断“专家是否学习互补特征”最重要的证据之一。

---

# 15. 可行性判定标准

以下是工程上的 go/no-go 规则，不是统计学定律。

## 15.1 继续研究 MoE 的最低条件

最终 E4 至少满足以下大部分条件：

1. OMI AUPRC 五折平均优于 E0。
2. OMI AUPRC 至少 4/5 folds 不劣于 E0，最好多数 folds 有正提升。
3. OMI F1 或 Sensitivity 有一致改善，而不是只提高 AUROC。
4. AMI AUROC/AUPRC 不出现明显下降。
5. Dual Gate 的 AMI/OMI 权重分布存在可观察差异。
6. 没有严重 Expert Collapse。
7. 参数量和推理成本仍可接受。

一个实用的初步“强正信号”可以定义为：

```text
OMI AUPRC mean improvement >= 1 percentage point
AND
AMI metric degradation < 0.5 percentage point
AND
>= 4 / 5 folds OMI AUPRC non-negative delta
```

如果提升小于该幅度但非常稳定，也不要直接否定，应结合置信区间和 OMI Sensitivity 判断。

## 15.2 停止继续堆复杂 MoE 的情况

如果出现：

```text
E2 <= E0
E3 <= E2
E4 <= E3
```

且 Gate 接近相同或 Expert 明显塌缩，则不要继续堆叠更多频率变体、KAN 或更多 Expert。

优先回到：

```text
Xception + Two Heads
```

重新分析任务是否真正存在可分离的专家偏好。

---

# 16. 计算成本控制

必须记录：

```text
Trainable Params
Total Params
FLOPs / MACs（如果现有项目有工具）
Peak GPU Memory
Training Time per Fold
Inference Time per ECG / batch
```

尤其比较：

```text
E0 vs E2 vs E3 vs E4
```

如果 E4 只获得极小提升但成本提高数倍，需要在最终结论中明确说明。

---

# 17. 实现要求

Codex 执行时：

1. **先扫描当前项目，不要假设目录结构。**
2. 找到：
   - dataset loader
   - label construction
   - current Xception model
   - training entrypoint
   - loss
   - evaluation code
   - CV / split logic
   - existing config system
3. 在现有架构上做最小侵入式改造。
4. 不重写整个训练框架。
5. 新模型建议模块化：

```text
models/
  xception_expert.*
  lead_gat_expert.*
  wavelnet_frequency_expert.*
  task_router.*
  kan_head.*
  ecg_hmoe_kan.*
```

实际目录按当前项目规范调整。

6. `forward()` 最好返回结构化结果：

```python
{
  "ami_logit": ...,
  "omi_logit": ...,
  "ami_gate": ...,
  "omi_gate": ...,
  "expert_features": {
      "morph": ...,
      "lead": ...,
      "freq": ...,
  }
}
```

便于后续分析。

7. 所有实验参数进入配置文件，不要散落 hardcode。
8. 不允许生成 MVMD/VMD 特征缓存；WavelNet 的母小波、kernel length、采样率等配置必须写入实验配置和 checkpoint metadata。
9. 所有异常、fallback 和未完成模块必须在最终报告明确说明，不允许静默跳过。

---

# 18. 必须输出的实验文件

至少生成：

```text
artifacts/
├── splits/
│   └── acs_ami_omi_5fold.csv
├── configs/
│   ├── e0_xception_two_heads.yaml
│   ├── e1_xception_kan.yaml
│   ├── e2_xception_mmoe.yaml
│   ├── e3_heterogeneous_mmoe.yaml
│   └── e4_hmoe_kan.yaml
├── predictions/
│   ├── e0_oof_predictions.csv
│   ├── e1_oof_predictions.csv
│   ├── e2_oof_predictions.csv
│   ├── e3_oof_predictions.csv
│   └── e4_oof_predictions.csv
├── gates/
│   └── e4_gate_weights.csv
├── metrics/
│   └── experiment_summary.csv
└── plots/
    ├── omi_pr_curves.png
    ├── ami_pr_curves.png
    ├── fold_metric_comparison.png
    ├── gate_weight_distribution.png
    └── expert_usage.png
```

命名可适配当前项目，但信息不能少。

---

# 19. Codex 最终必须生成的实验报告

创建：

```text
reports/acs_hmoe_kan_feasibility.md
```

报告必须包括：

## 19.1 Current Codebase

- 当前最佳 Xception 架构
- 当前数据 pipeline
- 当前 labels
- 当前训练方式
- 当前 baseline 指标

## 19.2 Dataset Audit

- ECG 数
- Patient 数
- AMI positives / negatives
- OMI positives / negatives
- 4 种联合标签组合
- 重复 ECG 患者统计
- 5-fold 标签分布

## 19.3 Implementation

逐模块描述：

- Xception Expert
- GAT Expert
- WavelNet Frequency Expert
- Projection
- AMI Router
- OMI Router
- KAN heads
- balance loss

## 19.4 Main Results

表格：

| Experiment | AMI AUROC | AMI AUPRC | AMI F1 | OMI AUROC | OMI AUPRC | OMI F1 | OMI Sens | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | | | | | | | | |
| E1 | | | | | | | | |
| E2 | | | | | | | | |
| E3 | | | | | | | | |
| E4 | | | | | | | | |

所有指标写五折：

```text
mean ± std
```

## 19.5 Per-fold Results

完整列出 5 folds，不得只写平均值。

## 19.6 Router Analysis

- AMI Router 平均权重
- OMI Router 平均权重
- AMI/OMI Gate divergence
- Expert Top-1 usage
- 是否存在 collapse

## 19.7 Ablation

如果 E4 有正结果，再输出消融结果。

## 19.8 Feasibility Conclusion

必须明确回答：

```text
1. Xception + MMoE 是否有效？
2. Heterogeneous Experts 是否比 homogeneous MLP experts 更好？
3. GAT 是否有贡献？
4. WavelNet Frequency Expert 是否有贡献？
5. KAN 是否优于 MLP Head？
6. AMI / OMI Router 是否学到了不同专家偏好？
7. 最终方案是否值得作为论文创新继续优化？
```

结论必须基于实验，不允许因为方案设计上“看起来合理”就判断可行。

---

# 20. 推荐执行顺序

```text
Step 1
扫描项目与数据集
      ↓
Step 2
固定 patient-level 5-fold split
      ↓
Step 3
复现 E0 baseline
      ↓
Step 4
运行 E1：KAN Head only
      ↓
Step 5
运行 E2：Homogeneous MMoE
      ↓
Step 6
实现 GAT + WavelNet Frequency Expert
      ↓
Step 7
运行 E3：Heterogeneous MMoE + MLP
      ↓
Step 8
运行 E4：Heterogeneous MMoE + KAN
      ↓
Step 9
比较五折结果
      ↓
Step 10
若 E4 有正增益，运行消融与 Router 分析
      ↓
Step 11
生成 feasibility report
```

---

# 21. 最重要的实验原则

请严格遵守：

```text
一次只验证一个主要变量。
```

不要直接：

```text
Baseline
→ Xception + GAT + WavelNet + MMoE + KAN + 新Loss
```

然后只报告一个最终结果。

必须通过：

```text
E0 → E1 → E2 → E3 → E4
```

逐步证明：

```text
KAN 是否有效
MMoE 是否有效
异构专家是否有效
频率专家是否有效
导联空间专家是否有效
最终组合是否值得继续
```

---

# 22. 第一阶段暂时不要做的内容

为避免模块堆砌，第一轮实验不要加入：

- Clinical-guided Router
- ECG-to-angiography time interval Router
- STEMI/NSTEMI auxiliary task
- Culprit vessel auxiliary task
- Top-K sparse routing
- Uncertainty estimation
- Contrastive learning
- PINN / physics loss
- Additional Transformer expert
- 多种小波母函数/尺度搜索（第一阶段固定 Symlet-4，不做此扩展）

如果 E4 明显有效，再从中选择 **一个** 方向作为下一阶段创新，不要全部同时加入。

---

# 23. 最终预期论文逻辑

如果实验成立，论文方法逻辑应当能够被概括为：

> 单一 ECG backbone 难以同时建模 12 导联 ECG 中互补的形态、导联空间和非平稳频率信息；同时，AMI 和 OMI 虽具有共享缺血表征，但其诊断目标并不完全一致。为此，构建由 Xception morphology expert、GAT lead-spatial expert 和端到端 WavelNet frequency expert 组成的异构专家池，并通过 AMI/OMI 任务特异性 Gate 对不同专家进行样本自适应融合，从而在共享知识的同时减少任务间负迁移。最终使用独立 KAN 分类器建模任务特异融合表征与 AMI/OMI 标签之间的复杂非线性关系。WavelNet 频率专家直接在网络第一层学习 Symlet-4 多尺度卷积，不进行 MVMD/VMD 外部预处理。

但只有在实验满足以下条件时才能保留这一论述：

```text
MMoE > Two-head baseline
Heterogeneous Experts > Homogeneous Experts
Dual Gate > Single Gate / Uniform Fusion
KAN > MLP Head（如果 KAN 被保留）
```

如果某个条件不成立，应删除对应模块，不要为了维持原设计而强行保留。

---

# 24. Codex 执行指令总结

请直接执行以下任务：

1. 扫描当前代码库，找到 ECG 数据、AMI/OMI 标签、Xception 最佳模型、训练与评估入口。
2. 不修改原始 baseline 行为，先复现并记录 E0。
3. 固定患者级 5-fold split，所有实验复用同一 split。
4. 依次实现并运行 E1、E2、E3、E4。
5. 所有实验沿用当前最佳训练策略，除新增模块外不随意调参。
6. 记录 AMI 与 OMI 每折 AUROC、AUPRC、F1、Sensitivity、Specificity。
7. OMI 重点关注 AUPRC、F1、Sensitivity。
8. 保存 OOF prediction、gate weights、expert usage 和所有配置。
9. 如果 E4 优于 baseline，再执行 Router/Expert 消融。
10. 最终生成 `reports/acs_hmoe_kan_feasibility.md`，明确判断方案：

```text
可行 / 部分可行 / 不建议继续
```

并用实验结果解释原因。
