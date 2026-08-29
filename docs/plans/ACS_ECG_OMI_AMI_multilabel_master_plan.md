# ACS ECG OMI/AMI 多标签分类总体研究方案

> **任务定义说明（2026-08-29）**：本文件保留早期 OMI/AMI 双标签方案
> 的历史设计。凡涉及 STEMI/NSTEMI/UA 的内容，不能将三个互斥亚型作为
> 独立 sigmoid；当前执行协议是 ACS `OTHER/STEMI/NSTEMI/UA` 单标签四分类
> 加独立 OMI/NOMI 二分类。请以
> [`CORRECTED_TASK_FORMULATION.md`](CORRECTED_TASK_FORMULATION.md) 及
> `experiments/acs_joint_*.py`、`configs/ecg_lfm/corrected_joint_*.yaml`
> 为准，旧章节仅用于追溯。

## 1. 研究目标

基于 ACS 12 导联 ECG 数据集，构建：

> **基于完整 10 秒 ECG 与深度时间序列模型的 OMI + AMI 双标签多标签分类系统。**

主要研究任务：

```text
Input
12-lead ECG
100 Hz × 10 sec
[1000, 12]
        │
        ▼
Proposed ECG Preprocessing
        │
        ▼
Time-Series Model
        │
        ▼
[OMI logit, AMI logit]
        │
        ▼
Asymmetric Loss
        │
        ▼
Sigmoid
        │
        ├── P(OMI)
        └── P(AMI)
```

标签定义：

```text
Y = [OMI, AMI]

OMI ∈ {0,1}
AMI ∈ {0,1}
```

这是：

```text
Multi-label Binary Classification
```

而不是 OMI 与 AMI 二选一分类。

---

# 2. 两个核心研究任务

整个实验体系分成两个核心任务。

## Task A：OMI + AMI 多标签分类

最终主要任务：

```text
ECG
 ↓
Shared Representation
 ↓
├── OMI
└── AMI
```

研究：

> 利用 AMI 辅助监督进行联合学习，是否能够改善 OMI 的 ECG 表征与识别性能。

---

## Task B：OMI 单标签二分类

另外建立：

```text
ECG
 ↓
OMI / Non-OMI
```

用于与原论文作者的 OMI baseline 直接比较。

这样形成：

```text
                    ECG
                     │
             Proposed Pipeline
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    TimesNet-OMI          TimesNet-MTL
      OMI only            OMI + AMI
          │                     │
          └──────────┬──────────┘
                     ▼
              Original Author
                 OMI CNN
```

---

# 3. 与原论文的区别

原论文 baseline：

```text
12-lead ECG
      ↓
Median waveform
      ↓
Multi-layer CNN
      ↓
OMI Binary Classification
```

原论文明确使用 12 导联 ECG 的 median waveform 作为 CNN 输入。

原论文测试集结果：

```text
Sensitivity = 0.697
Specificity = 0.873
PPV         = 0.277
NPV         = 0.976
Accuracy    = 0.861
F1          = 0.396
```



原论文训练使用：

```text
CrossEntropyLoss
Class weight [1,8]
Adam
Learning rate = 1e-4
Batch size = 32
Epoch = 70
Early stopping patience = 10
```



我们的方案改为：

```text
完整 10 秒 ECG
        ↓
ECG-specific preprocessing
        ↓
Deep Time-Series Model
        ↓
ASL
        ↓
OMI / AMI Joint Classification
```

主要创新点包括：

1. **完整 10 秒 ECG，而不是 median waveform**
2. **Amplitude-Preserving Global Robust Scaling**
3. **Morphology-Preserving ECG Augmentation**
4. **Asymmetric Loss 处理类别不平衡**
5. **OMI + AMI 联合多标签学习**
6. **采用现代深度时间序列分类模型进行系统比较**

---

# 4. 现有预处理数据

保留已有：

```text
prepare_acs_data.py
```

现有流程：

```text
Raw WFDB
500 Hz × 10 sec
        ↓
WFDB decode
        ↓
异常记录处理
        ↓
resample_poly
        ↓
100 Hz
        ↓
固定 1000 time points
```

原始 ECG：

```text
(5000, 12)
```

最终：

```text
(1000, 12)
```



当前已经生成：

```text
preprocessed_100hz/
├── unnormalized/
└── normalized/
```

其中 normalized 为：

```text
逐 ECG
+
逐导联
Min-Max → [0,1]
```



---

# 5. Proposed Preprocessing

最终方案主要使用：

```text
preprocessed_100hz/unnormalized/
```

继续处理：

```text
100 Hz unnormalized ECG
[1000,12]
        ↓
0.5–40 Hz
Zero-phase Butterworth
        ↓
Global Robust Scaling
        ↓
Morphology-Preserving Augmentation
        ↓
Model
```

---

# 6. Band-pass Filtering

使用：

```text
0.5–40 Hz
4th-order Butterworth
zero-phase
fs = 100 Hz
```

实现：

```python
scipy.signal.butter(..., output="sos")
scipy.signal.sosfiltfilt(...)
```

现有 `prepare_acs_data.py` 本身没有执行带通滤波，因此这是新的 Stage-B preprocessing。

---

# 7. Global Robust Scaling

只使用 Training Set 计算：

```text
global_median
global_MAD
```

定义：

```text
scale = 1.4826 × MAD
```

然后：

```text
X =
(X - global_median)
/
(scale + eps)
```

关键要求：

```text
所有 12 个导联
共享同一套 median + scale
```

避免逐导联归一化进一步破坏导联之间的相对幅值信息。

Validation / Test 只读取 Train scaler。

---

# 8. ECG Augmentation

只用于 Train。

## Gaussian Noise

```text
p = 0.3
SNR = 30–40 dB
```

## Global Amplitude Scaling

```text
p = 0.3
factor = 0.95–1.05
```

同一个 ECG 的 12 导联同步使用同一 factor。

## Temporal Shift

```text
p = 0.2
±0.2 sec
=
±20 samples @100 Hz
```

12 导联同步移动。

禁止主实验使用：

```text
SMOTE
MixUp
Lead Shuffle
强 Time Warping
```

---

# 9. Short ECG

两条异常记录：

```text
03228
14262
```

原始只有约：

```text
7 sec
```

当前处理方法是：

```text
700 samples
↓
Circular Repeat
↓
1000 samples
```



TimesNet 等时间序列模型主实验改为：

```text
700 valid samples
+
300 zero padding
+
padding mask
```

避免人为构造重复周期。

Circular Repeat 与 Padding + Mask 作为附加实验比较。

---

# 10. 数据划分

必须进行：

```text
Patient-level Leakage Audit
```

不能只检查：

```text
ecg_row_record
```

还必须保证：

```text
Patient_id
```

不存在跨 Train / Validation / Test 的情况。

如果官方 Test 标签可用：

```text
Official Train
├── Internal Train
└── Validation

Official Test
└── Final Test
```

如果官方 Test 标签隐藏：

```text
Official Train
├── Train
├── Validation
└── Internal Test
```

官方 Test 仅用于最终官方评价。

---

# 11. 主训练代码统一基于 Time-Series-Library

这是本项目的强制要求。

所有**新型时间序列模型**：

```text
TimesNet
Informer
Non-stationary Transformer
MambaSL
```

必须统一基于：

```text
https://github.com/thuml/Time-Series-Library
```

的代码结构、Dataset、训练框架和 classification pipeline 实现。

禁止：

```text
TimesNet 使用 Time-Series-Library

Informer 使用另外一个独立 repo

Non-stationary Transformer 使用另一个训练框架

MambaSL 再单独写一套训练代码
```

否则无法做到公平实验。

官方 Time-Series-Library 当前覆盖：

```text
classification
forecasting
imputation
anomaly detection
```

等任务，并已经将 TimesNet、Informer 和 Non-stationary Transformer 收录在统一模型库中。

MambaSL：

```text
MambaSL:
Exploring Single-Layer Mamba
for Time Series Classification
```

已被官方 Time-Series-Library 列为新加入的 classification baseline。

因此：

> **四个新型模型全部必须在 Time-Series-Library 统一 classification framework 下运行。**

---

# 12. Non-stationary Transformer 特别说明

论文原始代码仓库：

```text
https://github.com/thuml/Nonstationary_Transformers
```

但是：

**实验时不要单独运行这个仓库。**

官方 Nonstationary Transformer 仓库已经明确说明该模型后来被整合进：

```text
Time-Series-Library
```

并支持：

```text
classification
forecasting
imputation
anomaly detection
```

因此本项目必须优先使用：

```text
Time-Series-Library/models/
Nonstationary_Transformer.py
```

对应的 TSLib 实现。

独立仓库：

```text
thuml/Nonstationary_Transformers
```

只作为：

```text
论文来源
实现参考
```

而不是实验训练框架。

---

# 13. MambaSL 特别说明

MambaSL 是本项目中新加入的重要现代 baseline：

```text
MambaSL
Exploring Single-Layer Mamba
for Time Series Classification
ICLR 2026
```

Claude Code 必须首先检查当前 clone 的：

```text
Time-Series-Library
```

是否已经包含对应 MambaSL implementation。

如果当前 checkout 已包含：

```text
MambaSL
```

直接复用官方代码。

如果当前 checkout 因版本较旧未包含：

```text
不要自己凭空重新实现 MambaSL。
```

应：

1. 检查官方 Time-Series-Library 最新版本；
2. 找到官方 README 所引用的 MambaSL 实现；
3. 将官方实现同步到当前 TSLib 项目；
4. 仍使用本项目统一的：
   - Dataset
   - preprocessing
   - ASL
   - metrics
   - train/validation/test
   - threshold selection。

最终必须保证：

```text
MambaSL
```

运行在：

```text
Time-Series-Library experiment framework
```

中。

---

# 14. 所有模型统一修改为 Multi-label Classification

Time-Series-Library 原始 classification 主要按照普通分类任务运行。

本项目统一修改为：

```text
Output:
[B,2]
```

即：

```text
logits[:,0] = OMI
logits[:,1] = AMI
```

禁止：

```text
Softmax
Argmax
CrossEntropyLoss
```

统一：

```text
raw logits
    ↓
Asymmetric Loss
```

推理：

```text
raw logits
    ↓
sigmoid
    ↓
P(OMI), P(AMI)
```

---

# 15. 新型模型对比组

所有正式模型对比实验中，**新模型组必须至少包含以下四个模型：**

| ID | Model | Family | Code Source |
|---|---|---|---|
| N1 | **TimesNet** | Period/Frequency Modeling | THUML Time-Series-Library |
| N2 | **Informer** | Efficient Transformer | THUML Time-Series-Library |
| N3 | **Non-stationary Transformer** | Non-stationarity-aware Transformer | THUML Time-Series-Library |
| N4 | **MambaSL** | State Space / Mamba | THUML Time-Series-Library |

这四个模型不得缺失。

---

# 16. 传统基线模型

同时保留前面的经典模型：

| ID | Model | Family |
|---|---|---|
| B1 | LSTM | RNN |
| B2 | ResNet1D_Wang | CNN |
| B3 | InceptionTime | Multi-scale CNN |
| B4 | ViT-1D | Transformer |

这样形成：

```text
Traditional Baselines
│
├── LSTM
├── ResNet1D_Wang
├── InceptionTime
└── ViT
        │
        ▼
Modern Time-Series Models
│
├── TimesNet
├── Informer
├── Non-stationary Transformer
└── MambaSL
```

总共：

```text
8 个模型
```

---

# 17. 为什么这样组织模型

模型对比不按“随便挑几个模型”组织。

而是形成四条技术路线：

### RNN

```text
LSTM
```

验证传统 recurrent modeling。

### CNN

```text
ResNet1D_Wang
InceptionTime
```

验证局部、多尺度 ECG morphology 建模。

### Transformer

```text
ViT
Informer
Non-stationary Transformer
```

验证 Attention-based 时序建模。

### Modern General Time-Series Models

```text
TimesNet
MambaSL
```

验证：

```text
Period-based representation
vs
State-space sequence modeling
```

这样模型选择具有明确方法学逻辑。

---

# 18. 主模型仍然是 TimesNet

虽然新增多个现代 baseline：

```text
Informer
Non-stationary Transformer
MambaSL
```

但论文 Proposed Backbone 仍定义为：

```text
TimesNet
```

原因不是提前假设 TimesNet 一定最好。

而是：

> 本研究提出的是以 TimesNet 为核心的 OMI/AMI 多标签方法，需要通过完整模型对比实验验证这一选择。

如果最终实验发现：

```text
MambaSL > TimesNet
```

或其他模型显著优于 TimesNet：

不能删除结果。

必须如实报告，并重新判断：

```text
TimesNet 是否仍适合作为 Proposed Backbone
```

---

# 19. 模型对比公平性

全部 8 个模型统一：

```text
Same Patient Split
Same ECG Input
Same Sampling Rate
Same Filtering
Same Scaling
Same Augmentation
Same ASL
Same Threshold Strategy
Same Metrics
Same Seeds
```

即统一：

```text
100 Hz ECG
      ↓
0.5–40 Hz
      ↓
Global Robust Scaling
      ↓
Train-only Augmentation
      ↓
Model
      ↓
ASL
```

只改变：

```text
Backbone
```

---

# 20. 模型比较表

最终生成：

| Model | Type | OMI AUPRC | OMI AUROC | AMI AUPRC | AMI AUROC | Macro AUPRC | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| LSTM | RNN | | | | | | |
| ResNet1D_Wang | CNN | | | | | | |
| InceptionTime | CNN | | | | | | |
| ViT | Transformer | | | | | | |
| Informer | Transformer | | | | | | |
| Non-stationary Transformer | Transformer | | | | | | |
| MambaSL | State Space | | | | | | |
| **TimesNet** | Period-based | | | | | | |

同时记录：

```text
Parameters
Training Time / Epoch
Inference Time
Peak GPU Memory
```

---

# 21. 原论文 OMI 二分类实验

建立：

```text
TimesNet-OMI
```

只预测：

```text
OMI
```

输出：

```text
[B,1]
```

使用完全相同 Proposed preprocessing。

形成：

| Method | Input | Task | Sens | Spec | PPV | NPV | Accuracy | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Author CNN | Median waveform | OMI | 0.697 | 0.873 | 0.277 | 0.976 | 0.861 | 0.396 |
| TimesNet-OMI | Full ECG | OMI | | | | | | |
| **TimesNet-MTL** | **Full ECG** | **OMI+AMI** | | | | | | |

这张表负责回答两个问题：

```text
Author CNN
vs
TimesNet-OMI
```

验证新的：

```text
完整 ECG
+
Preprocessing
+
TimesNet
+
ASL
```

是否改善 OMI。

然后：

```text
TimesNet-OMI
vs
TimesNet-MTL
```

验证：

```text
AMI auxiliary supervision
```

是否进一步提高 OMI。

---

# 22. 新模型也做 OMI-only 还是只做多标签？

正式方案中：

## 所有 8 个模型

必须做：

```text
OMI + AMI
```

主多标签模型对比。

即：

```text
LSTM-MTL
ResNet1D-MTL
InceptionTime-MTL
ViT-MTL
Informer-MTL
NonstationaryTransformer-MTL
MambaSL-MTL
TimesNet-MTL
```

---

## 原论文 OMI Benchmark

重点只需要：

```text
Author CNN
TimesNet-OMI
TimesNet-MTL
```

不需要把全部 8 个模型再训练一遍 OMI-only。

否则实验数量过大，而且不能增加主要论证价值。

---

# 23. Loss 对比

固定：

```text
TimesNet
```

比较：

```text
BCEWithLogitsLoss
vs
ASL
```

回答：

> ASL 是否改善 OMI/AMI 类别不平衡？

---

# 24. Preprocessing 对比

固定：

```text
TimesNet + ASL
```

比较：

### P0 Existing

```text
100 Hz
+
Per-record Per-lead Min-Max
```

### P1

```text
100 Hz
+
Global Robust
```

### P2

```text
100 Hz
+
0.5–40 Hz
+
Global Robust
```

### P3 Proposed

```text
100 Hz
+
0.5–40 Hz
+
Global Robust
+
Augmentation
```

---

# 25. 消融实验

Full：

```text
Band-pass
+
Global Robust
+
Augmentation
+
TimesNet
+
ASL
+
OMI/AMI Joint Learning
```

消融：

| ID | Filter | Robust | Aug | ASL | Multi-label |
|---|---|---|---|---|---|
| A0 Full | ✓ | ✓ | ✓ | ✓ | ✓ |
| A1 | ✓ | ✓ | ✓ | BCE | ✓ |
| A2 | ✓ | Min-Max | ✓ | ✓ | ✓ |
| A3 | ✓ | ✓ | × | ✓ | ✓ |
| A4 | × | ✓ | ✓ | ✓ | ✓ |
| A5 | ✓ | ✓ | ✓ | ✓ | OMI-only |

---

# 26. 完整实验体系

最终论文实验体系：

```text
Experiment 1
Original Author OMI Benchmark
│
├── Author CNN
├── TimesNet-OMI
└── TimesNet-MTL


Experiment 2
Multi-label Model Comparison
│
├── Traditional
│   ├── LSTM
│   ├── ResNet1D_Wang
│   ├── InceptionTime
│   └── ViT
│
└── Modern TSLib Models
    ├── TimesNet
    ├── Informer
    ├── Non-stationary Transformer
    └── MambaSL


Experiment 3
Loss Comparison
│
├── BCE
└── ASL


Experiment 4
Preprocessing Comparison
│
├── Existing Min-Max
├── Global Robust
├── Filter + Robust
└── Proposed


Experiment 5
Ablation Study
│
├── w/o ASL
├── w/o Global Robust
├── w/o Augmentation
├── w/o Filtering
└── OMI-only


Experiment 6
Short ECG Strategy
│
├── Circular Repeat
└── Padding + Mask
```

---

# 27. Claude Code 对 Time-Series-Library 的实施要求

Claude Code 开始开发前必须先检查当前：

```text
Time-Series-Library/
```

版本。

重点检查：

```text
models/TimesNet.py
models/Informer.py
models/Nonstationary_Transformer.py
MambaSL implementation
```

不要根据文件名假设实现存在。

---

## 已存在

直接复用官方：

```text
Time-Series-Library
```

实现。

只改：

```text
Multi-label classification adaptation
```

不要重写 backbone。

---

## 不存在

尤其 MambaSL 如果当前本地 checkout 较旧：

不要自己设计一个“类似 MambaSL”的模型。

必须：

```text
官方 Time-Series-Library
        ↓
确认最新实现
        ↓
同步官方代码
        ↓
接入当前 TSLib framework
```

---

# 28. 统一输出接口

所有模型最终统一：

```text
input:
[B,1000,12]

output:
[B,2]
```

输出：

```text
[:,0] = OMI logit
[:,1] = AMI logit
```

Loss：

```text
ASL
```

Evaluation：

```text
Sigmoid
↓
label-specific validation thresholds
```

---

# 29. 随机种子

正式模型对比：

```text
2026
2027
2028
2029
2030
```

全部 8 个模型运行相同 5 seeds。

最终：

```text
mean ± std
```

禁止：

```text
TimesNet 五次
其他模型一次
```

---

# 30. 最终需要回答的研究问题

整个实验最终必须回答：

### RQ1

完整 10 秒 ECG + TimesNet 是否优于原论文 median-waveform CNN 的 OMI 二分类结果？

### RQ2

OMI + AMI 联合学习是否优于 OMI 单任务？

### RQ3

TimesNet 与：

```text
Informer
Non-stationary Transformer
MambaSL
```

这些 Time-Series-Library 新型模型相比表现如何？

### RQ4

TimesNet 与：

```text
LSTM
ResNet1D_Wang
InceptionTime
ViT
```

传统基线相比表现如何？

### RQ5

ASL 是否改善类别不平衡？

### RQ6

Global Robust Scaling、Filtering 和 ECG Augmentation 各自贡献多少？

---

# 31. 最终论文方法定位

最终 Proposed Pipeline：

```text
Raw 12-lead ECG
500 Hz × 10 sec
        │
        ▼
prepare_acs_data.py
        │
        ▼
100 Hz
[1000,12]
        │
        ▼
0.5–40 Hz
        │
        ▼
Global Robust Scaling
        │
        ▼
Morphology-Preserving Augmentation
        │
        ▼
TimesNet
        │
        ▼
Shared Representation
      ┌────┴────┐
      ▼         ▼
     OMI       AMI
      │         │
      └────┬────┘
           ▼
          ASL
           │
           ▼
     OMI + AMI
Multi-label Prediction
```

核心论证路线：

```text
原论文 CNN
      ↓
TimesNet-OMI
      ↓
TimesNet-MTL
      ↓
与传统模型比较
      ↓
与现代 TSLib 模型比较
      ↓
Loss / Preprocessing / Ablation
      ↓
验证 Proposed Method
```

其中现代模型比较必须明确包含：

```text
TimesNet
Informer
Non-stationary Transformer
MambaSL
```

并且**全部基于 THUML Time-Series-Library 的统一代码体系完成实验**。
