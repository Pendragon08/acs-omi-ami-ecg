# ECG-LFM 四标签微调实施方案（Codex 最终执行版）

> **目标**：基于 ECG-LFM 官方公开预训练权重与 fine-tuning 代码，在当前 ACS 数据集上完成  
> `STEMI / NSTEMI / UA / OMI` 四标签多输出预测，并与现有 `E4 HMoE-KAN` 做严格公平对比。  
> **用途**：本 README 可直接交给 Codex，要求基于官方 ECG-LFM 仓库实现一个独立实验分支，不破坏当前 E4 代码与结果。  
> **核心原则**：先复现官方 ECG-LFM fine-tuning，再适配当前四标签任务；不要一开始把 ECG-LFM 与 E4 的 GAT/WavelNet/HMoE 混在一起。

---

# 0. 已确认的数据集来源与事实（Codex 必须以此为准）

本实验使用的数据集**确定为以下 Scientific Data 2026 论文公开的数据集**：

**Title**  
A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome prediction containing 19,955 ECGs

**Journal**  
Scientific Data, 2026

**DOI**  
`10.1038/s41597-026-07278-0`

**Nature article**  
https://www.nature.com/articles/s41597-026-07278-0

**Official Figshare dataset**  
https://doi.org/10.6084/m9.figshare.29925314

**Dataset paper code / validation repository**  
https://github.com/catcatgirl/validation-code

已确认的关键数据事实：

```text
Total ECGs:        19,955
Patients:          18,909
Leads:             12
Lead set:          I, II, III, aVR, aVL, aVF, V1–V6
Duration:          10 s
Original sampling: 500 Hz
Raw format:        WFDB (.dat + .hea)
Median waveform:   .med
```

当前研究使用的四个目标标签：

```text
STEMI
NSTEMI
UA
OMI
```

其中：

```text
STEMI / NSTEMI / UA
```

在当前任务标签定义中按原数据表作为三个独立 binary outputs 训练，但原始临床标签彼此互斥；

```text
OMI
```

是独立的闭塞标签，可与 STEMI 或 NSTEMI 同时为阳性，因此最终模型必须使用：

```text
4 independent sigmoid outputs
```

而不是四分类 Softmax。

## 0.1 当前 100 Hz P2 的定位

当前 E4 使用的：

```text
P2
0.5–40 Hz
100 Hz
12 leads
10 s
```

是**已经预处理和降采样后的实验数据**。

它继续用于：

```text
E4 HMoE-KAN
```

复现实验，但**不得作为 ECG-LFM 主实验的原始输入**。

ECG-LFM 主实验必须：

```text
Scientific Data official Figshare
→ 重新下载原始 500 Hz WFDB
→ 单独建立 ECG-LFM preprocessing
→ fine-tuning
```

禁止：

```text
P2 100 Hz
→ interpolation / resample to 500 Hz
→ 当作原始 ECG-LFM input
```

因为上采样无法恢复降采样时已经丢失的信息。

## 0.2 数据重新下载后不得重新划分 cohort

重新下载原始波形只改变：

```text
input waveform source
```

不能改变：

```text
record IDs
patient IDs
labels
train/external partition
5-fold assignments
```

必须继续复用当前实验定义：

```text
Training pool: 17,960 ECGs
Patients:      17,018
External test: 1,995 ECGs
5-fold:        patient-level iterative multilabel stratified
Seed:          2026
```

因此 Codex 的关键工作不是重新构造一个新 cohort，而是：

```text
official raw WFDB waveform
        ↓ record_id join
existing metadata / labels / patient_id / split / fold
```

如果官方原始数据和当前 metadata 无法一一对齐，必须先生成 discrepancy report，**禁止自动丢样本后直接训练**。

---

# 1. 研究问题

当前 E4 HMoE-KAN 已经完成：

```text
12-lead ECG
→ Xception morphology expert
→ LeadSpatial GAT expert
→ WavelNet frequency expert
→ task-specific routing
→ 4 KAN heads
→ STEMI / NSTEMI / UA / OMI
```

当前 OOF 结果：

```text
Macro AUROC = 0.8208 ± 0.0049
Macro AUPRC = 0.4503 ± 0.0076
Macro F1    = 0.4147 ± 0.0256
```

当前主要问题：

1. NSTEMI PPV/F1 较低；
2. NSTEMI 与 UA 尚未得到可靠区分；
3. OMI 排序能力尚可，但当前阈值下 PPV 偏低；
4. 当前训练集仅约 1.8 万份 ECG，远小于大型 ECG foundation model 的预训练规模。

因此本实验要回答：

> **大规模自监督 ECG foundation model 的通用 ECG 表征，是否比当前从本数据集监督学习得到的 E4 表征更适合 STEMI / NSTEMI / UA / OMI 四标签预测？**

---

# 2. 主要参考论文：ECG-LFM

## 2.1 论文信息

**Title**  
A self-supervised electrocardiogram foundation model for empowering cardiovascular disease prediction and genetic factor discovery

**Journal**  
Nature Communications

**Year**  
2026

**Volume / Article**  
Volume 17, Article 5771

**DOI**  
`10.1038/s41467-026-72436-2`

**Paper**  
https://www.nature.com/articles/s41467-026-72436-2

**Nature Communications 2025 JIF**  
`18.1`

官方期刊指标：  
https://www.nature.com/ncomms/journal-impact

---

## 2.2 ECG-LFM 为什么适合本任务

ECG-LFM 在：

```text
11,571,587 份 12-lead ECG
1,979,599 individuals
```

上进行自监督预训练。

预训练数据主要来自：

```text
Harvard-Emory ECG Database
MIMIC-IV-ECG
```

模型不是依赖疾病标签训练，而是先学习 ECG 自身的通用 representation。

原论文 fine-tune 后用于多种 CVD：

```text
MI
ST/T change
conduction disturbance
hypertrophy
ventricular ectopics
AF
...
```

其跨多个下游数据集平均 AUROC 约：

```text
0.930
```

论文还专门测试小样本 fine-tuning。即使只使用 PTB-XL 训练集的一小部分，ECG-LFM 仍显著优于若干监督模型。

这与本任务非常匹配，因为：

```text
STEMI / NSTEMI / OMI 正类样本有限
```

尤其：

```text
NSTEMI
```

属于当前最困难标签之一。

---

# 3. ECG-LFM 的核心方法

ECG-LFM 不是普通 CNN。

总体：

```text
12-lead ECG
    ↓
Convolutional Feature Encoder
    ↓
Latent ECG Tokens
    ↓
Quantization Module
    ↓
Random Masking
    ↓
Conformer Context Encoder
    ↓
Contextual ECG Representations
```

预训练阶段同时使用三种自监督目标：

```text
1. Contextual Contrastive Loss
2. Masked Language Modeling Loss
3. Multi-segment Contrastive Loss
```

总预训练损失：

```math
L = L_CC + L_MLM + L_MC
```

三个 loss 的权重原论文均设为 1。

---

## 3.1 Contextual Contrastive Learning

让模型根据周围 ECG 上下文恢复被 Mask 的 latent representation。

目的：

```text
学习局部 ECG pattern
+
学习上下文关系
```

---

## 3.2 Masked Modeling

类似 BERT：

```text
ECG token sequence
→ mask approximately part of tokens
→ model predicts masked token IDs
```

用于学习长距离 ECG 依赖。

---

## 3.3 Multi-segment Contrastive Learning

同一 ECG 的两个连续 segment：

```text
Segment A
Segment B
```

应该产生相近 representation。

不同患者/不同 ECG：

```text
representation should be separated
```

目的：

> 提高 patient-level ECG representation consistency。

---

# 4. 官方代码与预训练权重

## 4.1 官方 GitHub

https://github.com/biomed-AI/ECG-LFM

仓库包含：

```text
examples/
fairseq_cli/
fairseq_signals/
scripts/preprocess/
README.md
```

ECG-LFM 基于：

```text
fairseq_signals
+
Hydra
+
PyTorch
```

---

## 4.2 官方预训练权重

Zenodo：

https://zenodo.org/records/20388950

文件：

```text
pretrain.pt
```

大小：

```text
约 7.5 GB
```

MD5：

```text
afa8dfc680590948f82aabbdb123bf43
```

Zenodo DOI：

```text
10.5281/zenodo.20388950
```

建议下载：

```bash
wget -O pretrain.pt \
  "https://zenodo.org/records/20388950/files/pretrain.pt?download=1"
```

然后校验：

```bash
md5sum pretrain.pt
```

应得到：

```text
afa8dfc680590948f82aabbdb123bf43
```

---

## 4.3 官方 Fine-tuning 配置

官方使用：

```text
examples/ECGLFM/config/finetuning/diagnosis.yaml
```

地址：

https://github.com/biomed-AI/ECG-LFM/blob/master/examples/ECGLFM/config/finetuning/diagnosis.yaml

Raw：

https://raw.githubusercontent.com/biomed-AI/ECG-LFM/master/examples/ECGLFM/config/finetuning/diagnosis.yaml

该配置明确使用：

```yaml
task:
  _name: ecg_classification

criterion:
  _name: binary_cross_entropy_with_logits

model:
  _name: ecg_transformer_classifier
  model_path: ???
  num_labels: 1
  encoder_embed_dim: 1024
```

因此官方模型本身已经支持：

```text
multi-label BCEWithLogits
```

本任务只需要：

```text
num_labels = 4
```

即可适配：

```text
STEMI
NSTEMI
UA
OMI
```

---

# 5. 官方 Fine-tuning 命令

官方 README 示例：

```bash
FAIRSEQ_SIGNALS_ROOT="<TODO>"
PRETRAINED_MODEL="<TODO>"
MANIFEST_DIR="<TODO>"
OUTPUT_DIR="<TODO>"
NUM_LABELS=4

fairseq-hydra-train \
    task.data=$MANIFEST_DIR \
    model.model_path=$PRETRAINED_MODEL \
    model.num_labels=$NUM_LABELS \
    optimization.lr=[1e-06] \
    optimization.max_epoch=100 \
    dataset.batch_size=256 \
    dataset.num_workers=5 \
    dataset.disable_validation=true \
    distributed_training.distributed_world_size=1 \
    distributed_training.find_unused_parameters=True \
    checkpoint.save_dir=$OUTPUT_DIR \
    checkpoint.save_interval=1 \
    checkpoint.keep_last_epochs=0 \
    common.log_format=csv \
    --config-dir $FAIRSEQ_SIGNALS_ROOT/examples/ECGLFM/config/finetuning/ \
    --config-name diagnosis
```

---

# 6. 官方论文 Fine-tuning 参数

Nature Communications 论文 Methods：

```text
Optimizer        Adam
Learning rate    1e-6
Batch size       256
Max epochs       100
Adam beta1       0.9
Adam beta2       0.98
Adam eps         1e-8
Early stopping   patience 10
Checkpoint       validation loss
Scheduler        exponential scheduler
```

注意：

> 官方 GitHub `diagnosis.yaml` 当前默认配置和论文 Methods 并非每个字段完全一致。

例如 raw config 中：

```yaml
optimization.lr: 5e-5
lr_scheduler: fixed
```

但官方 README 的 fine-tune 命令显式覆盖：

```text
lr = 1e-6
```

论文则写：

```text
exponential scheduler
```

### Codex 实施原则

第一版应优先复现：

```text
论文 Methods + 官方 README command
```

即：

```text
lr = 1e-6
max_epoch = 100
batch = 256（显存不足可下降）
Adam
early stopping = validation loss / patience 10
```

Scheduler：

```text
优先实现 paper-faithful exponential scheduler
```

同时保留：

```text
fixed scheduler
```

配置用于复现官方当前 repo config。

不要无记录地混用。

---

# 7. 一个非常关键的问题：当前 P2 输入与 ECG-LFM 原始输入不一致

当前 E4 使用：

```text
P2 ECG
0.5–40 Hz
100 Hz
12 leads
10 s
```

但 ECG-LFM 原论文使用的预处理是：

```text
resample to 500 Hz
high-pass 1 Hz
low-pass 30 Hz (2nd-order Butterworth)
notch 50/60 Hz
10 s
z-score normalization
split into non-overlapping 5 s segments
```

原论文数据大部分本身也是：

```text
10 s
500 Hz
12 leads
```

而当前 Scientific Data 数据论文确认原始 ECG：

```text
10 s
500 Hz
12 leads
```

因此：

> **不要直接把当前 P2 的 100 Hz 信号输入 ECG-LFM，然后把结果称为“官方 ECG-LFM fine-tuning”。**

这是一个明显 input-domain mismatch。

---

# 8. 原始 500 Hz 数据必须重新从官方 Figshare 下载

当前项目中的：

```text
P2 ECG
0.5–40 Hz
100 Hz
12 leads
10 s
```

是已经处理过的数据。

**本 ECG-LFM 实验禁止把 P2 作为主输入，也禁止把 100 Hz P2 简单上采样回 500 Hz。**

原因：

```text
100 Hz processed ECG
→ 已经过滤 / 降采样
→ 高频和原始采样细节已经丢失

100 → 500 Hz interpolation
≠
原始 500 Hz ECG
```

因此 Codex 必须从该 Scientific Data 论文的**官方数据源重新下载原始 ECG**，建立一条与现有 P2 完全独立的数据分支。

---

## 8.1 官方原始数据地址

Scientific Data 论文明确给出的 Data Records / Data Availability：

**Figshare DOI**

```text
https://doi.org/10.6084/m9.figshare.29925314
```

论文页面：

```text
https://www.nature.com/articles/s41597-026-07278-0
```

论文说明公开数据包含三个主要组成部分：

```text
1. raw ECG recordings
2. ECG median data
3. diagnostic documentation
```

其中原始 ECG waveform 使用 **WFDB format**：

```text
<record>.dat   # binary raw waveform
<record>.hea   # WFDB header
```

median waveform：

```text
<record>.med
```

本实验必须使用：

```text
.dat + .hea
```

对应的 **raw ECG recordings**。

不要使用 `.med` 代替 raw ECG。

---

## 8.2 原始 ECG 官方规格

论文明确说明：

```text
12 leads:
I, II, III, aVR, aVL, aVF, V1–V6

sampling rate:
500 Hz

duration:
10 s
```

因此理论形状约为：

```text
12 × 5000 samples
```

实际加载时以 `.hea` header 中记录的：

```text
fs
sig_len
n_sig
lead names
units
gain
```

为准，不要硬编码假设后直接训练。

---

## 8.3 重新下载，而不是复用当前本地缓存

Codex 必须将这次 ECG-LFM 所需原始数据下载到一个新的独立目录，例如：

```text
data/
  raw_acs_figshare_500hz/
```

现有：

```text
P2 / processed / 100hz
```

目录保持不动。

禁止：

```text
覆盖 P2
删除旧数据
修改旧 E4 输入文件
```

这样可以保证：

```text
E4 experiment
和
ECG-LFM experiment
```

相互独立且可复现。

---

## 8.4 下载实现原则

Codex 不要根据猜测硬编码 Figshare 文件名。

应先解析 Figshare article / DOI 对应的实际文件列表，然后下载所有原始数据相关文件。

优先方案：

```text
Figshare DOI
→ resolve article
→ inspect file list
→ identify raw ECG archive/files
→ download
```

如果 Figshare 提供 archive：

```text
download archive
→ checksum / size check
→ extract
```

如果提供多个文件：

```text
逐个下载
→ retry
→ size validation
```

下载脚本建议新增：

```text
scripts/download_acs_raw_figshare.py
```

至少支持：

```text
--output-dir
--resume
--verify
--skip-existing
```

要求：

```text
HTTP retry
partial download resume（若实现方便）
download log
file size validation
```

不要默默跳过下载失败文件。

---

## 8.5 数据下载后必须生成 manifest

下载并解压后扫描所有：

```text
*.hea
*.dat
```

确保一一配对。

生成：

```text
artifacts/raw_500hz_manifest.csv
```

字段至少包括：

```text
record_id
hea_path
dat_path
sampling_rate
signal_length
num_leads
lead_names
units
file_size_dat
file_size_hea
load_ok
```

同时输出：

```text
artifacts/raw_500hz_download_report.json
```

至少记录：

```text
source DOI
download date
number of files
number of WFDB records
number of valid pairs
number of invalid/missing pairs
sampling-rate distribution
signal-length distribution
lead-count distribution
```

---

## 8.6 必须使用 WFDB 正确读取原始信号

推荐 Python：

```python
import wfdb

record = wfdb.rdrecord(record_base_path)
signal = record.p_signal
fs = record.fs
sig_name = record.sig_name
```

优先使用：

```text
p_signal
```

得到物理单位下的波形，而不是直接把 `.dat` bytes 当数字读。

加载后检查：

```text
shape
NaN / Inf
sampling rate
lead order
units
gain
baseline
```

---

## 8.7 必须统一 12 导联顺序

目标固定：

```text
I
II
III
aVR
aVL
aVF
V1
V2
V3
V4
V5
V6
```

如果 WFDB header 中顺序不同：

```text
按 sig_name 重排
```

如果缺导联：

```text
不要静默补零
```

先记录并报告数量。

默认第一版：

```text
缺任何必要导联 → exclude from ECG-LFM experiment
```

除非原论文/数据实际表明不存在此问题。

---

## 8.8 原始数据必须重新与当前标签表对齐

不要通过文件排序对齐。

必须通过稳定的：

```text
ECG record ID
```

把：

```text
raw WFDB waveform
```

与：

```text
STEMI
NSTEMI
UA
OMI
patient_id
train/external split
fold assignment
```

做显式 join。

生成：

```text
artifacts/ecg_lfm_master_index.csv
```

至少包含：

```text
record_id
patient_id
raw_hea_path
raw_dat_path

STEMI
NSTEMI
UA
OMI

dataset_split
fold
```

Join 后必须检查：

```text
duplicate record IDs
missing labels
missing waveform
unexpected extra raw records
```

---

## 8.9 必须保留当前 external test 划分

重新下载原始数据不意味着重新划分数据。

仍然必须保持当前实验：

```text
Training pool:
17,960 ECG
17,018 patients

External test:
1,995 ECG
```

如果原始官方数据中包含全部 19,955 ECG：

```text
只通过已有 record ID / split file
恢复当前 train / external 分组
```

禁止重新随机拆成：

```text
90/10
```

或调用官方 ECG-LFM 自带的随机 split。

---

## 8.10 五折也必须保持与 E4 完全一致

在 17,960 条 training pool 上：

```text
patient-level
iterative multilabel stratified
5 folds
seed = 2026
```

必须复用当前保存好的 fold assignments。

禁止：

```text
重新跑一个“差不多”的五折
```

否则：

```text
E4 vs ECG-LFM
```

无法做严格 paired comparison。

---

## 8.11 下载后先核对官方数据规模

最终 Master Index 应解释：

```text
官方 raw records
↓
成功匹配当前 19,955 metadata
↓
训练 17,960
外测 1,995
```

如果数量不一致：

```text
停止完整训练
```

先输出 discrepancy report。

Codex 不能为了凑数量：

```text
随意删除
自动换用 P2
```

---

## 8.12 官方 dataset preparation code

Scientific Data 论文同时提供数据准备 / 验证代码：

```text
https://github.com/catcatgirl/validation-code.git
```

用途：

- 了解作者的数据组织；
- 核对 waveform / metadata；
- 核对 validation 逻辑。

Codex 可以参考，但本项目自己的：

```text
patient split
labels
ECG-LFM preprocessing
```

仍以当前实验定义为准。

---

## 8.13 推荐下载与预处理流水线

```text
Official Figshare
https://doi.org/10.6084/m9.figshare.29925314
          │
          ▼
Download raw ECG package/files
          │
          ▼
Validate .hea/.dat pairs
          │
          ▼
WFDB read
          │
          ▼
Check:
500 Hz
12 leads
10 s
          │
          ▼
Join current labels + patient IDs + splits
          │
          ▼
raw_500hz_master_index.csv
          │
          ▼
ECG-LFM native preprocessing
          │
          ▼
1–30 Hz
notch
z-score
          │
          ▼
2 × 5s segments
          │
          ▼
ECG-LFM fine-tuning
```

---

## 8.14 本步骤必须先于任何 ECG-LFM 五折训练

完整执行顺序：

```text
Step 0
下载 raw official dataset

Step 1
WFDB integrity audit

Step 2
record / label / patient / split alignment audit

Step 3
ECG-LFM preprocessing smoke test

Step 4
single-record model forward

Step 5
small-subset overfit

Step 6
Fold 0

Step 7
full 5-fold
```

只要 Step 0–3 中任何一个失败：

```text
禁止直接启动 GPU 五折
```

---

# 9. 推荐 ECG-LFM 数据输入方案

## 8.1 Main experiment：paper-compatible 500 Hz

如果当前数据仓中可以访问原始 500 Hz ECG：

**必须优先使用这一版本。**

流程：

```text
Raw ECG 500 Hz
      ↓
ECG-LFM preprocessing
      ↓
1 Hz high-pass
30 Hz low-pass
50/60 Hz notch
      ↓
z-score normalization
      ↓
10 s / 5000 points
      ↓
2 × non-overlap 5 s segments
      ↓
ECG-LFM
```

这是主实验。

---

## 8.2 不要做假的 100 → 500 Hz“恢复”

如果只有：

```text
100 Hz P2
```

不能简单：

```text
100 Hz
→ interpolate to 500 Hz
```

然后声称等价于原始 500 Hz。

上采样无法恢复已经丢失的高时间分辨率信息。

可以作为：

```text
P2-adapted ECG-LFM
```

补充实验，但不能替代主实验。

---

# 10. 5 秒 segment 如何变回每份 10 秒 ECG 的 prediction

ECG-LFM 原论文把 10 秒 ECG 分成：

```text
segment 1 = first 5 s
segment 2 = second 5 s
```

本任务最终需要：

```text
one prediction per ECG
```

推荐：

```text
segment1 → logits1
segment2 → logits2

record_logit =
(logits1 + logits2) / 2

record_probability =
sigmoid(record_logit)
```

不要把两个 5 秒 segment 当成最终两条独立测试记录。

所有 metrics / threshold / platform output 均按：

```text
10 s original ECG record
```

聚合。

需要保存：

```text
segment-level logits
record-level logits
record-level probabilities
```

便于检查。

---

# 11. 四标签定义

固定 label order：

```text
0 STEMI
1 NSTEMI
2 UA
3 OMI
```

标签：

```text
y ∈ {0,1}^4
```

例如：

```text
non-ACS NOMI = [0,0,0,0]

STEMI + OMI  = [1,0,0,1]

NSTEMI + OMI = [0,1,0,1]

UA            = [0,0,1,0]
```

STEMI / NSTEMI / UA 在现有标注中互斥。

OMI：

```text
can overlap with STEMI or NSTEMI
```

因此模型使用：

```text
4 independent sigmoid outputs
```

而不是 softmax。

---

# 12. Loss

为了与当前 E4 做公平比较：

第一轮 ECG-LFM 必须使用：

```text
BCEWithLogitsLoss
unweighted
```

与 E4 保持一致。

即：

```math
L =
BCE(y_STEMI, z_STEMI)
+
BCE(y_NSTEMI, z_NSTEMI)
+
BCE(y_UA, z_UA)
+
BCE(y_OMI, z_OMI)
```

官方 `diagnosis.yaml` 也是：

```text
binary_cross_entropy_with_logits
```

如果官方 criterion 强制要求：

```text
weights_file
```

则第一版为公平比较，应使用：

```text
all-one weights
```

而不是自动计算 class weights。

后续再做：

```text
unweighted BCE
vs
pos_weight BCE
vs
Asymmetric Loss
```

独立消融。

---

# 13. 数据划分必须复用当前 E4 的 exact split

当前：

```text
Train pool = 17,960 ECG
17,018 patients

External test = 1,995 ECG
```

已有：

```text
patient-level iterative multilabel stratified 5-fold
seed = 2026
```

ECG-LFM **禁止重新随机切五折**。

必须直接读取现有：

```text
fold assignments
patient IDs
record IDs
```

保证：

```text
E4 和 ECG-LFM
完全使用同一组 5 folds
```

否则对比没有意义。

---

# 14. 五折实现

对于每个 Fold：

```text
Fold k
│
├── train patients
├── validation patients
└── no access to external test labels
```

每折：

```text
1. preprocess ECG-LFM 500 Hz input
2. train / fine-tune from same pretrain.pt
3. checkpoint by validation loss
4. generate validation probabilities
5. generate held-out fold probabilities
6. store OOF probabilities
7. generate external-test probabilities
```

五折结束：

```text
OOF = concatenation of 5 held-out fold predictions
```

External:

```text
p_external =
mean(
p_fold1,
p_fold2,
p_fold3,
p_fold4,
p_fold5
)
```

---

# 15. Threshold 必须重新计算

当前 E4 的 threshold：

```text
STEMI  0.06
NSTEMI 0.04
UA     0.35
OMI    0.03
```

这些 threshold **不能直接用于 ECG-LFM**。

因为：

```text
ECG-LFM probability calibration
!=
E4 probability calibration
```

所以 ECG-LFM 必须重新：

```text
OOF probabilities
→ 0.01–0.99 grid
→ Youden J
→ 4 frozen thresholds
```

再应用 external test。

---

# 16. Threshold 评价原则

由于阈值调优会影响：

```text
F1
PPV
Sensitivity
Specificity
```

但不会改变：

```text
AUROC
AUPRC
```

因此 E4 vs ECG-LFM 首要公平比较：

```text
Macro AUROC
Macro AUPRC

per-label AUROC
per-label AUPRC
```

然后再比较各自 OOF 学出的最优 threshold 下：

```text
F1
Sensitivity
Specificity
PPV
NPV
```

不要用：

```text
E4 threshold
```

测试 ECG-LFM。

---

# 17. 重点分析 NSTEMI vs UA

这是本项目目前最大的 clinical / modeling bottleneck。

必须从 OOF predictions 中取：

```text
ground truth == NSTEMI
or
ground truth == UA
```

单独评价：

```text
NSTEMI vs UA AUROC
NSTEMI vs UA AUPRC
Macro F1
Balanced Accuracy
NSTEMI sensitivity
UA sensitivity
Confusion matrix
```

目的：

> 判断大规模自监督 ECG 表征是否真的改善了目前 E4 最弱的 discriminative boundary。

---

# 18. 第一轮必须做的三种 ECG-LFM adaptation

不能只跑一个设置。

---

## L0：Linear Probe

```text
freeze ECG-LFM encoder
↓
train only 4-output classification layer
```

作用：

> 测试 ECG-LFM 原始 pretrained embedding 是否已经包含 STEMI/NSTEMI/UA/OMI 信息。

---

## L1：Official-style Fine-tuning

```text
load pretrain.pt
↓
ECG-LFM classifier
↓
fine-tune according to official diagnosis pipeline
```

这是主方案。

---

## L2：Fine-tune + Unfreeze Ablation

如果 L1 中：

```text
feature_grad_mult = 0
```

冻结部分 convolutional encoder，

再增加可选实验：

```text
feature_grad_mult > 0
```

或 gradual unfreeze。

目的：

> 判断 ACS domain 是否需要更多 low-level adaptation。

第一轮不要直接把所有层大 LR 全量训练，避免破坏 pretrained representation。

---

# 19. ECG-LFM 与 E4 的核心对比矩阵

| ID | Model | Pretraining | Main architecture | 目的 |
|---|---|---|---|---|
| B0 | Xception baseline | No | Xception | 基础监督模型 |
| B1 | E4 HMoE-KAN | No external SSL | Heterogeneous MoE | 当前最好方案 |
| L0 | ECG-LFM Linear Probe | 11.57M ECG SSL | Frozen ECG-LFM | 预训练 embedding 能力 |
| L1 | **ECG-LFM Fine-tune** | 11.57M ECG SSL | ECG-LFM | **主新方案** |
| L2 | ECG-LFM enhanced unfreeze | 11.57M ECG SSL | ECG-LFM | 域适配消融 |

---

# 20. 成功标准

ECG-LFM 值得保留，如果：

### Overall

```text
Macro AUPRC > E4
```

或至少：

```text
Macro AUROC 稳定提升
```

### NSTEMI

重点：

```text
NSTEMI AUPRC ↑
NSTEMI F1 ↑
```

### NSTEMI vs UA

至少：

```text
pairwise AUROC / AUPRC / Balanced Accuracy
```

出现稳定改善。

### OMI

不能因为提高 NSTEMI 而显著降低：

```text
OMI AUROC / AUPRC
```

---

# 21. 如果 ECG-LFM 比 E4 更强，下一步怎么做

不要马上：

```text
ECG-LFM + GAT + WavelNet + HMoE + KAN
```

全部合并。

应该按顺序：

```text
Step 1
ECG-LFM alone
vs E4

Step 2
ECG-LFM + KAN head
vs ECG-LFM linear/MLP head

Step 3
ECG-LFM + auxiliary culprit-vessel task
```

每次只增加一个变量。

---

# 22. 如果 ECG-LFM 对 NSTEMI 仍没有明显提升

这也是重要结果。

说明：

> NSTEMI vs UA 的瓶颈可能并不是模型基础 ECG representation 不够好，而是 ECG 本身对 myocardial necrosis 的信息有限。

因为 NSTEMI / UA 的临床区别依赖：

```text
troponin / myocardial injury evidence
```

而当前模型只有 ECG。

这时不要继续无限堆复杂 backbone。

应转向：

```text
clinical knowledge distillation
or
multimodal lab + ECG
```

而不是继续扩大 ECG-only 模型。

---

# 23. 与当前 P2 的公平性问题

当前 E4 使用：

```text
100 Hz
0.5–40 Hz
robust scaler
```

ECG-LFM paper-native 使用：

```text
500 Hz
1–30 Hz
z-score
5 s segments
```

这两个模型的 preprocessing 不同。

### 正确的论文比较方式

不是：

> 强制所有模型使用完全相同 preprocessing。

而是：

> 每个模型首先使用其论文原生/最合理 preprocessing，并固定相同 patient splits 和 labels。

同时可以增加一个辅助实验：

```text
ECG-LFM-P2
```

即 ECG-LFM 在当前 P2 input 上适配。

但是主结果应优先使用：

```text
ECG-LFM-native preprocessing
```

否则可能低估 pretrained checkpoint。

---

# 24. Codex 推荐代码结构

不要修改 E4 现有模型。

新增独立目录：

```text
experiments/
  ecg_lfm/

models/
  ecg_lfm_adapter.py

data/
  ecg_lfm_preprocess.py
  ecg_lfm_manifest.py

configs/
  ecg_lfm/
    linear_probe.yaml
    finetune.yaml

scripts/
  download_ecg_lfm.sh
  run_ecg_lfm_5fold.sh
  evaluate_ecg_lfm.py
  ensemble_ecg_lfm.py
```

若当前项目结构不同，Codex 先扫描现有代码再映射。

---

# 25. 推荐第三方代码管理方式

不要把 ECG-LFM 官方仓库代码大规模复制到当前 source tree。

推荐：

```text
third_party/
  ECG-LFM/
```

或者：

```text
external/
  ECG-LFM/
```

保持官方 commit 可追踪。

Codex 必须记录：

```text
repo URL
commit SHA
checkpoint MD5
```

用于论文复现。

---

# 26. 安装

官方建议 Python 3.9：

```bash
git clone https://github.com/biomed-AI/ECG-LFM.git
cd ECG-LFM

conda create -n ecg-lfm python=3.9 -y
conda activate ecg-lfm

pip install --editable ./
```

官方 README 标注：

```text
PyTorch >= 1.5
Python >= 3.6
```

论文实际实现环境：

```text
Python 3.9.20
PyTorch 2.5.1
Ubuntu 22.04.2
```

建议 Codex 优先建立：

```text
Python 3.9
current compatible PyTorch/CUDA
```

不要因为 README 的最低版本强行安装很旧 PyTorch。

---

# 27. 数据 Manifest

官方代码通过：

```text
fairseq_signals
```

manifest 驱动数据读取。

官方脚本：

```bash
python fairseq_signals/data/ecg/preprocess/manifest.py \
    /path/to/data \
    --dest /path/to/manifest
```

但当前项目已有固定 patient-level folds。

因此 Codex **不要使用官方随机 `--test-percent/--valid-percent` 重新切数据**。

应：

```text
读取当前 fold CSV / patient IDs
↓
针对每个 fold 自己生成 train / valid manifest
```

保证 split 完全一致。

---

# 28. 标签适配

必须创建：

```text
label_def.csv
```

固定顺序：

```text
STEMI
NSTEMI
UA
OMI
```

和对应四列 binary target。

在写入官方 task 格式之前：

```text
先 inspect ECG-LFM / fairseq_signals ecg_classification task
```

不要猜 manifest / label binary 格式。

Codex 应复用官方 parser 所要求的格式。

---

# 29. Criterion 配置注意

官方 `diagnosis.yaml`：

```yaml
criterion:
  _name: binary_cross_entropy_with_logits
  report_auc: true
  weights_file: ???
```

当前 E4 使用：

```text
unweighted BCEWithLogitsLoss
```

因此第一轮公平实验：

```text
weights_file = all ones
```

或修改 wrapper 让其 equivalent to unweighted BCE。

必须做一个小 synthetic unit test：

```text
official criterion output
==
torch.nn.BCEWithLogitsLoss output
```

允许浮点误差。

---

# 30. 模型输出

官方 config：

```yaml
model:
  _name: ecg_transformer_classifier
  num_labels: 1
  encoder_embed_dim: 1024
```

本任务：

```yaml
model:
  num_labels: 4
```

必须输出：

```text
[B, 4]
```

label order 必须全流程固定。

---

# 31. Checkpoint selection

当前 E4：

```text
validation macro-AUROC
```

选择 checkpoint。

ECG-LFM 原论文：

```text
validation loss
```

选 checkpoint。

为了同时实现：

### Experiment L1a：paper-faithful

```text
select by validation loss
```

### Experiment L1b：E4-fair

```text
select by validation macro-AUROC
```

先跑 L1a。

资源允许再 L1b。

不要只选最好的方式而不报告选择标准。

---

# 32. Early stopping

Paper:

```text
patience = 10
```

当前 E4：

```text
patience = 15
```

ECG-LFM 主复现使用：

```text
10
```

然后公平性消融可以：

```text
15
```

但不是必须第一轮就跑。

---

# 33. Batch size

原论文：

```text
256
```

如果 GPU 无法承受：

```text
batch 32 / 64 / 128
+
gradient accumulation
```

使：

```text
effective batch size ≈ 256
```

记录实际：

```text
physical batch
gradient accumulation
effective batch
```

---

# 34. 外部测试规则

External test：

```text
1,995 ECG
```

严格：

```text
never train
never validate
never threshold tune
never checkpoint select
```

五折 ECG-LFM：

```text
5 models
→ external probabilities
→ mean ensemble
→ frozen OOF-derived thresholds
→ final prediction
```

---

# 35. 输出文件

每 Fold：

```text
fold_k/
  best.pt
  train_log.csv
  val_metrics.json
  oof_predictions.csv
  external_predictions.csv
```

汇总：

```text
ecg_lfm_cv_summary.csv
ecg_lfm_oof.csv
ecg_lfm_thresholds.json
ecg_lfm_external_ensemble.csv
ecg_lfm_vs_e4.csv
ecg_lfm_nstemi_vs_ua.csv
```

---

# 36. 必须保存的 prediction 字段

```text
patient_id
ecg_id

y_STEMI
y_NSTEMI
y_UA
y_OMI

p_STEMI
p_NSTEMI
p_UA
p_OMI

fold
```

External test 如果无 GT：

```text
只保存 probabilities + predicted labels
```

---

# 37. Unit tests

Codex 必须加：

### Input

```text
[B,12,2500]  # 5s × 500 Hz
```

模型能够 forward。

### Output

```text
[B,4]
```

### Segment aggregation

两 segment：

```text
[B,2,4]
```

正确聚合成：

```text
[B,4]
```

### Label order

确保：

```text
STEMI,NSTEMI,UA,OMI
```

始终一致。

### Split leakage

```text
train/valid patient IDs no overlap
```

### External leakage

external IDs 不出现在 train folds。

### BCE equivalence

官方 criterion 和 unweighted PyTorch BCE 对照。

---

# 38. 推荐第一轮实验顺序

不要直接一次全跑五折再发现数据接口错误。

## Stage 0：Environment

```text
clone repo
download checkpoint
verify MD5
```

## Stage 1：Single-record inference smoke test

一条 500 Hz ECG：

```text
preprocess
→ 2 × 5s
→ ECG-LFM
→ [2,4]
→ aggregate
→ [4]
```

## Stage 2：Small subset overfit

例如：

```text
64–256 training ECG
```

验证模型能够下降 loss。

## Stage 3：Fold 0

完整训练一个 fold。

检查：

```text
AUROC
AUPRC
prediction distribution
no NaN
```

## Stage 4：Full 5-fold

只有 Fold 0 正常后才跑。

---

# 39. 最重要的公平比较

最终报告：

| Metric | Xception | E4 HMoE-KAN | ECG-LFM Linear Probe | ECG-LFM Fine-tune |
|---|---:|---:|---:|---:|
| Macro AUROC | | | | |
| Macro AUPRC | | | | |
| Macro F1 | | | | |
| STEMI AUPRC | | | | |
| NSTEMI AUPRC | | | | |
| UA AUPRC | | | | |
| OMI AUPRC | | | | |
| NSTEMI vs UA AUROC | | | | |
| NSTEMI vs UA F1 | | | | |

还要报告：

```text
Params
Trainable Params
GPU memory
Training time
Inference time
```

---

# 40. 论文中可以形成的研究问题

如果 ECG-LFM 显著优于 E4：

> Large-scale self-supervised ECG representation learning provides stronger transferable representations for ACS and OMI prediction than task-specific supervised heterogeneous expert learning on a relatively limited cohort.

如果 E4 更好：

> Task-specific heterogeneous ECG representations may outperform general-purpose foundation representations for fine-grained ACS subtype and OMI discrimination.

无论哪种结果，都具有研究意义。

---

# 41. 第二篇方法依据：ST-MEM（辅助参考）

**Title**  
Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electrocardiogram

**Conference**  
ICLR 2024 Main Conference

**Paper**  
https://openreview.net/forum?id=WcOohbsF4H

**Official code**  
https://github.com/vuno/ST-MEM

ST-MEM 提出：

```text
12-lead ECG
→ lead × temporal patches
→ random masking
→ Transformer encoder
→ reconstruct masked patches
```

它直接证明：

> ECG 自监督 masked representation learning 能够学习 12 导联的 spatio-temporal relationships，并在 labeled ECG 较少时提高下游任务表现。

因此 ECG-LFM 的自监督迁移路线不仅有 Nature Communications 依据，也有 ICLR 顶会 ECG SSL 方法依据。

---

# 42. 第三篇任务定义参考

**Development and validation of machine learning algorithms based on electrocardiograms for cardiovascular diagnoses at the population level**

**Journal**  
npj Digital Medicine, 2024

**DOI**  
`10.1038/s41746-024-01130-8`

**Paper**  
https://www.nature.com/articles/s41746-024-01130-8

**JIF 2025**  
`18.0`

用途：

> 支持在全人群 ECG 上同时输出多个 independent binary cardiovascular diagnosis probabilities。

这与当前：

```text
STEMI
NSTEMI
UA
OMI
```

四 sigmoid 设计一致。

---

# 43. 当前数据集论文

**Title**  
A large-scale 12-lead electrocardiogram dataset for acute coronary syndrome prediction containing 19,955 ECGs

**Journal**  
Scientific Data, 2026

**Paper**  
https://www.nature.com/articles/s41597-026-07278-0

**DOI**  
`10.1038/s41597-026-07278-0`

数据论文确认：

```text
raw ECG = 12 leads
sampling = 500 Hz
duration ≈ 10 s
```

因此原始数据本身适合建立 ECG-LFM native preprocessing 分支。

---

# 44. 官方资源清单

## ECG-LFM Nature Communications paper

https://www.nature.com/articles/s41467-026-72436-2

## ECG-LFM GitHub

https://github.com/biomed-AI/ECG-LFM

## ECG-LFM checkpoint

https://zenodo.org/records/20388950

## Fine-tuning config

https://github.com/biomed-AI/ECG-LFM/blob/master/examples/ECGLFM/config/finetuning/diagnosis.yaml

## Fine-tuning config raw

https://raw.githubusercontent.com/biomed-AI/ECG-LFM/master/examples/ECGLFM/config/finetuning/diagnosis.yaml

## ST-MEM paper

https://openreview.net/forum?id=WcOohbsF4H

## ST-MEM code

https://github.com/vuno/ST-MEM

## Current ACS dataset

https://www.nature.com/articles/s41597-026-07278-0

---

# 45. 给 Codex 的最终执行要求

> 注：以下编号以执行先后为准；新增原始数据下载步骤后，Codex 应按依赖顺序执行，不要仅依赖旧编号。

请基于当前项目实现 ECG-LFM 四标签对比实验，遵循以下要求：

1. **不要修改或删除当前 E4 HMoE-KAN。**
2. 新建独立 `ecg_lfm` experiment/model path。
3. 从 ACS 数据集官方 Figshare `https://doi.org/10.6084/m9.figshare.29925314` 重新下载原始 500 Hz WFDB `.dat/.hea` 数据；禁止以当前 100 Hz P2 上采样代替。
4. 为原始数据生成完整 download/integrity manifest，并按 record_id 与当前 patient/label/split 表显式对齐。
5. 下载官方 ECG-LFM `pretrain.pt` 并校验 MD5。
6. 记录 ECG-LFM Git commit SHA、ACS 数据 DOI、下载日期和数据清单。
7. 读取当前固定的 patient-level 5-fold split；禁止重新随机划分。
6. 外部 1,995 ECG 继续保持完全隔离。
7. 主实验优先使用原始 500 Hz ECG，而不是当前 100 Hz P2。
8. 按 ECG-LFM 论文实现：
   - 500 Hz
   - high-pass 1 Hz
   - low-pass 30 Hz
   - notch 50/60 Hz
   - z-score
   - 10 s
   - 2 × 5 s segment
9. label order 固定：
   ```text
   STEMI,NSTEMI,UA,OMI
   ```
10. `model.num_labels=4`。
11. 第一轮 loss 使用 **unweighted BCEWithLogitsLoss**，与 E4 公平比较。
12. 先实现 linear probe，再实现官方 style fine-tuning。
13. Fine-tuning 第一版：
   ```text
   lr 1e-6
   max epochs 100
   batch effective 256
   Adam beta=(0.9,0.98)
   early stopping 10
   ```
14. checkpoint 主复现按 validation loss。
15. 每个原始 ECG 的两个 5s segment 输出需聚合回一条 record prediction。
16. 五折输出 OOF probabilities。
17. ECG-LFM 独立在 OOF 上重新搜索 Youden J threshold；禁止复用 E4 threshold。
18. 五折模型 ensemble 后预测 external test。
19. 重点计算：
   - Macro AUROC/AUPRC/F1
   - 每个标签 AUROC/AUPRC/F1
   - NSTEMI vs UA pairwise metrics
20. 与 E4 完全同表比较。
21. 输出模型参数量、trainable params、显存、训练时间、推理时间。
22. 先跑 single-record smoke test，再 small subset overfit，再 Fold 0，最后才能启动完整 5-fold。
23. 不要在第一轮同时加入 GAT/WavelNet/MoE/KAN；本实验目标是单独验证 ECG-LFM foundation representation。
24. 最终提交：
   - 修改文件列表
   - 运行命令
   - 5-fold summary
   - thresholds
   - external predictions
   - ECG-LFM vs E4 comparison
   - NSTEMI vs UA analysis
   - preprocessing compatibility说明

---

# 46. 推荐 Codex 最终结论模板

完成实验后只根据数据得出结论。

如果 ECG-LFM 更好：

```text
ECG-LFM pretrained representation significantly/stably improves ...
```

如果无提升：

```text
Large-scale generic ECG pretraining did not yield consistent gains over
the task-specific E4 HMoE-KAN framework on this ACS cohort.
```

如果只有 NSTEMI 提升：

```text
The main benefit of foundation-model transfer is concentrated in the
label-scarce NSTEMI task...
```

禁止在结果出来之前预设：

```text
ECG-LFM 一定优于 E4
```

---

# 47. 最终方案一句话

> **直接使用 Nature Communications 2026 开源 ECG-LFM 的 1157 万 ECG 自监督预训练权重，在与当前 E4 完全相同的患者级五折和外部测试划分上，对 STEMI、NSTEMI、UA、OMI 四个独立标签进行官方风格 fine-tuning，重新学习模型自身的 OOF 分类阈值，并重点检验其是否能改善目前最困难的 NSTEMI 以及 NSTEMI-vs-UA 区分。**

# 验收门禁：Codex 在完整五折训练前必须全部通过

以下门禁未通过时，不允许启动完整 5-fold GPU 训练。

```text
[ ] 1. 官方 Figshare 原始数据下载完成
[ ] 2. raw .dat/.hea 成对完整
[ ] 3. WFDB 可正常读取
[ ] 4. 绝大多数/全部目标记录为 500 Hz、12-lead、约 10 s；任何异常已报告
[ ] 5. lead names 已按 I,II,III,aVR,aVL,aVF,V1–V6 对齐
[ ] 6. raw record_id 与当前 19,955 metadata 显式 join
[ ] 7. 17,960 train + 1,995 external split 与现有实验完全一致
[ ] 8. patient-level 5-fold assignment 与 E4 完全一致
[ ] 9. external-test patient/record 不进入训练、验证、阈值搜索
[ ] 10. ECG-LFM pretrain.pt 下载并通过 MD5 校验
[ ] 11. ECG-LFM Git commit SHA 已记录
[ ] 12. single-record forward 输出 shape=[1,4]
[ ] 13. 两个 5 s segment 能聚合回单个 10 s record prediction
[ ] 14. small-subset overfit 成功，loss 能正常下降
[ ] 15. Fold 0 训练稳定，无 NaN/Inf，指标与概率分布合理
```

完整五折结束后必须提交：

```text
1. raw data download/integrity report
2. master record alignment table
3. exact fold/split verification
4. training config
5. ECG-LFM commit SHA
6. pretrain checkpoint MD5
7. five-fold metrics
8. OOF predictions
9. OOF-derived four thresholds
10. external-test ensemble predictions
11. ECG-LFM vs E4 comparison table
12. NSTEMI-vs-UA dedicated analysis
13. trainable parameters / GPU memory / training time / inference time
```

只有满足这些条件，ECG-LFM 与 E4 HMoE-KAN 的比较才可以用于论文结论。
