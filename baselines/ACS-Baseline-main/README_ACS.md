# ACS 心电二分类基线对比实验

## 1. 任务简介

本任务使用 ACS（Acute Coronary Syndrome，急性冠脉综合征）数据中的 12 导联心电信号，分别完成以下两个二分类任务：

- **AMI**：急性心肌梗死（Acute Myocardial Infarction）识别；
- **OMI**：闭塞性心肌梗死（Occlusion Myocardial Infarction）识别。

两个任务相互独立。标签 `1` 表示患病，标签 `0` 表示不患病。训练脚本会针对 AMI 和 OMI 分别初始化并训练模型，不将二者作为一个多标签任务联合训练。

主训练脚本为：

```text
main_train_acs_baselines.py
```

默认会用 8 种基线模型分别训练 AMI 和 OMI，共运行 16 组实验。

## 2. 数据集目录

相关数据位于：

```text
data/ACS/
├── CSV/
│   └── train.csv
└── preprocessed_100hz/
    └── normalized/
        ├── train/
        │   ├── 04904.npy
        │   └── ...
        └── test/
```

当前实验**只读取**以下 NPY 目录：

```text
data/ACS/preprocessed_100hz/normalized/train
```

当前实验**只读取**以下标签文件：

```text
data/ACS/CSV/train.csv
```

`normalized/test` 目录暂时不会被读取、划分或评估。模型只根据训练数据内部划分出的验证集选择最佳权重。

## 3. 信号和标签格式

### 3.1 心电信号

每个 `.npy` 文件对应一条 12 导联心电记录。预期形状为：

```text
(1000, 12)
```

含义为：

- 100 Hz 采样率；
- 1000 个时间点；
- 12 个 ECG 导联；
- 每条记录长度为 10 秒；
- 信号已经完成逐记录、逐导联 Min-Max 归一化。

数据加载时会将信号转置为 PyTorch 模型使用的格式：

```text
(12, 1000)
```

批次输入形状为：

```text
(batch_size, 12, 1000)
```

脚本也兼容形状已经是 `(12, 1000)` 的文件，但其他形状会触发错误，以避免错误数据进入训练过程。

### 3.2 CSV 标签

`train.csv` 至少需要包含以下字段：

```text
ecg_row_record,AMI,OMI
```

匹配方式如下：

```text
ecg_row_record = 04904.dat
去除扩展名后  = 04904
对应信号文件  = 04904.npy
```

脚本会检查：

- NPY 和 CSV 是否成功匹配；
- CSV 中是否有重复记录；
- `AMI` 和 `OMI` 是否只包含 `0` 和 `1`；
- NPY 与 CSV 各自未匹配的记录数量。

当前数据检查结果为：

```text
NPY 文件：17960
CSV 记录：17960
成功匹配：17960
```

标签分布为：

```text
AMI：0 = 15281，1 = 2679
OMI：0 = 16809，1 = 1151
```

## 4. 数据集划分

每个任务都根据自身标签进行分层随机划分：

```text
80% 训练集
20% 验证集
```

默认随机种子为 `10`。使用 `train_test_split(..., stratify=labels)`，因此训练集和验证集会尽可能保持原始正负样本比例。

默认划分结果为：

| 任务 | 集合 | 总数 | 阴性 | 阳性 |
|---|---|---:|---:|---:|
| AMI | 训练集 | 14368 | 12225 | 2143 |
| AMI | 验证集 | 3592 | 3056 | 536 |
| OMI | 训练集 | 14368 | 13447 | 921 |
| OMI | 验证集 | 3592 | 3362 | 230 |

测试集暂时不参与训练、验证或模型选择。

## 5. 默认基线模型

默认对比以下模型：

1. `resnet1d_wang`
2. `xresnet1d101`
3. `inceptiontime`
4. `fcn_wang`
5. `lstm`
6. `lstm_bidir`
7. `vit`
8. `mobilenetv3_small`

这些模型均封装在 `models/` 目录中，并通过 `models/__init__.py` 对外暴露模型构造函数。

所有模型必须支持以下调用形式：

```python
model = model_constructor(num_classes=1)
```

并接收以下输入：

```text
(batch_size, 12, 1000)
```

普通二分类模型应输出：

```text
(batch_size, 1)
```

训练使用：

- `BCEWithLogitsLoss`；
- Adam 优化器；
- 默认学习率 `0.001`；
- 默认批次大小 `64`；
- 默认训练 `100` 个 epoch；
- 训练阶段添加标准差为 `0.1` 的高斯噪声，验证阶段不添加噪声。

## 6. 运行方式

进入项目目录：

```bash
cd "/mnt/y/new_20260812/数据预处理/MVMS-net-main"
```

### 6.1 运行全部实验

运行 8 个模型的 AMI 和 OMI 实验：

```bash
python main_train_acs_baselines.py
```

默认共运行 16 组实验。

### 6.2 正式训练前进行快速检查

建议先运行一个模型、一个任务和一个 epoch：

```bash
python main_train_acs_baselines.py \
  --models resnet1d_wang \
  --targets AMI \
  --epochs 1 \
  --batch-size 64 \
  --device cuda:0
```

### 6.3 只运行指定任务

只训练 AMI：

```bash
conda run -n py311 python main_train_acs_baselines.py --targets AMI
```

只训练 OMI：

```bash
conda run -n py311 python main_train_acs_baselines.py --targets OMI
```

### 6.4 只运行指定模型

```bash
conda run -n py311 python main_train_acs_baselines.py \
  --models resnet1d_wang inceptiontime fcn_wang
```

也可以同时限制任务：

```bash
conda run -n py311 python main_train_acs_baselines.py \
  --models lstm lstm_bidir \
  --targets OMI
```

### 6.5 修改训练参数

```bash
conda run -n py311 python main_train_acs_baselines.py \
  --batch-size 128 \
  --epochs 50 \
  --lr 0.0005 \
  --seed 10 \
  --workers 4 \
  --device cuda:0
```

主要参数如下：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--data-dir` | ACS normalized train 目录 | NPY 输入目录 |
| `--csv-path` | ACS `train.csv` | 标签 CSV |
| `--checkpoint-dir` | `checkpoints/` | 最佳权重目录 |
| `--result-dir` | `results/` | 实验结果目录 |
| `--models` | 全部 8 个模型 | 要运行的模型 |
| `--targets` | `AMI OMI` | 要运行的任务 |
| `--batch-size` | 64 | 批次大小 |
| `--epochs` | 100 | 每组实验的 epoch 数 |
| `--lr` | 0.001 | Adam 学习率 |
| `--seed` | 10 | 数据划分及训练随机种子 |
| `--workers` | 0 | DataLoader 工作进程数 |
| `--device` | 自动选择 CUDA/CPU | 训练设备 |

## 7. 权重与结果文件

### 7.1 最佳模型权重

保存目录：

```text
checkpoints/
```

命名规则：

```text
{model_name}_acs_{target}_checkpoint_best.pth
```

例如：

```text
checkpoints/resnet1d_wang_acs_ami_checkpoint_best.pth
checkpoints/resnet1d_wang_acs_omi_checkpoint_best.pth
```

权重仅在验证集 AUC 提升时覆盖，因此最终保留的是验证集 AUC 最佳的 epoch，而不是最后一个 epoch。

权重内容包括：

```text
model_state_dict
optimizer_state_dict
global_epoch
best_auc
model_name
target
```

### 7.2 逐 epoch 结果

保存目录：

```text
results/
```

命名规则：

```text
{model_name}_acs_{target}_result.csv
```

记录训练集和验证集的：

- Loss；
- Accuracy；
- F1；
- AUC；
- TPR（敏感度/召回率）；
- TNR（特异度）；
- 截至当前 epoch 的最佳验证 AUC。

### 7.3 最佳结果摘要

命名规则：

```text
{model_name}_acs_{target}_best.json
```

其中记录最佳 epoch、最佳验证指标以及训练/验证集的正负样本数量。

## 8. 引入更多模型

新增模型时，应先将模型完整封装到 `models/` 目录中，再由主脚本统一调用。不要把模型结构直接写进 `main_train_acs_baselines.py`。

### 8.1 在 `models/` 中创建模型文件

例如新增 `new_ecg_model`，创建：

```text
models/new_ecg_model.py
```

模型构造函数应接受 `num_classes`，示例：

```python
import torch
from torch import nn


class NewECGModel(nn.Module):
    def __init__(self, num_classes: int = 1, input_channels: int = 12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).squeeze(-1)
        return self.classifier(x)


def new_ecg_model(num_classes: int = 1) -> NewECGModel:
    return NewECGModel(num_classes=num_classes)
```

必须满足：

- 输入为 `[batch_size, 12, 1000]`；
- 输出为 `[batch_size, num_classes]`；
- ACS 二分类中 `num_classes=1`；
- 返回的是原始 logits，不要在模型末尾手动调用 sigmoid；
- 模型构造函数能够通过 `num_classes=1` 创建模型。

训练脚本会在计算指标时调用 sigmoid，而损失函数 `BCEWithLogitsLoss` 已经在内部包含稳定的 sigmoid 运算。

### 8.2 在 `models/__init__.py` 中导出模型

添加：

```python
from .new_ecg_model import new_ecg_model
```

这样主脚本才能通过以下方式找到模型：

```python
getattr(models, model_name)(num_classes=1)
```

### 8.3 在主脚本中注册模型名称

打开 `main_train_acs_baselines.py`，将名称加入 `MODEL_NAMES`：

```python
MODEL_NAMES = (
    "resnet1d_wang",
    "xresnet1d101",
    "inceptiontime",
    "fcn_wang",
    "lstm",
    "lstm_bidir",
    "vit",
    "mobilenetv3_small",
    "new_ecg_model",
)
```

注册后可以单独运行：

```bash
conda run -n py311 python main_train_acs_baselines.py \
  --models new_ecg_model \
  --targets AMI OMI
```

如果不传 `--models`，新增模型会和列表中的其他模型一起运行。

### 8.4 新模型接入检查

正式训练前，建议先检查模型输入输出：

```python
import torch
import models

model = models.new_ecg_model(num_classes=1)
x = torch.randn(2, 12, 1000)
y = model(x)
print(y.shape)
```

预期输出：

```text
torch.Size([2, 1])
```

然后执行一轮小规模训练：

```bash
conda run -n py311 python main_train_acs_baselines.py \
  --models new_ecg_model \
  --targets AMI \
  --epochs 1 \
  --batch-size 8
```

确认数据加载、前向传播、反向传播、权重保存和结果保存都正常后，再运行完整实验。

## 9. 注意事项

1. OMI 阳性样本比例较低，应优先关注 AUC、F1、TPR 和 TNR，而不应只看 Accuracy。
2. 当前脚本按验证 AUC选择最佳模型，没有使用测试数据。
3. 所有模型使用同一个随机种子和相同的分层划分规则，便于公平比较。
4. 改变 `--seed` 会同时影响训练/验证划分和模型初始化。
5. 使用多个 DataLoader worker 时可以提高读取速度，但 WSL 或内存有限环境下建议先从 `--workers 0` 开始。
6. 运行完整 16 组实验耗时较长，建议先完成单模型、单任务、单 epoch 测试。
7. 如果新增模型有特殊输入格式，应优先在模型内部完成适配，保持主训练脚本提供的统一输入 `[batch, 12, 1000]`。
