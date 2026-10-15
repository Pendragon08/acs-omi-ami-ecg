# ACS–OMI Joint-6: ECG Classification

<div align="center">

**基于 ECG-LFM 的 ACS–OMI 联合六分类研究**

[实验报告](docs/reports/实验结果总报告_最终.md) · [模型对比](docs/ECG_LFM_Joint6_CV_Comparison_Report.md) · [结果数据](results/joint6_cv/summary/) · [文档导航](docs/README.md)

</div>

本仓库保存 ACS–OMI 心电图分类实验的代码、配置、聚合指标和研究记录。主任务使用 **ECG-LFM + LoRA + 单一六分类 Softmax**，从联合类别概率计算 ACS 分类与 OMI 概率，并通过患者级嵌套五折交叉验证比较不同模型。

```text
12 导联 ECG → 预处理与患者级划分 → ECG-LFM / 对照模型
           → Joint-6 联合分类 → ACS / OMI 概率 → 冻结阈值评估
```

## Project Status

- [x] 完成 7 种模型 × 5 个外层折的主对比，保存逐折指标与汇总。
- [x] 归档标签审计、评价修订、训练曲线及未取得收益的探索实验。
- [ ] 当前 Joint-6 方法的独立外部测试尚无已归档结果。

以下成绩来自已归档实验。历史报告中的“训练中”是当时的进度记录，不表示当前仍有训练任务。

## Table of Contents

1. [Setup](#setup)
2. [Phase 1: Data and Protocol](#phase-1-data-and-protocol)
3. [Phase 2: Training](#phase-2-training)
4. [Phase 3: Evaluation](#phase-3-evaluation)
5. [Results](#results)
6. [Repository Structure](#repository-structure)
7. [Documentation](#documentation)
8. [Acknowledgements and License](#acknowledgements-and-license)

## Setup

```bash
git clone https://github.com/Pendragon08/acs-omi-ami-ecg.git
cd acs-omi-ami-ecg
```

复现实验使用 Linux、Python 3.9、PyTorch 2.5.1 / CUDA 12.1 和 ECG-LFM / Fairseq。完整依赖版本及安装顺序见 [环境配置](configs/ecg_lfm/env-spec.json)。本仓库没有统一的根目录 `requirements.txt`。

准备上述环境后，安装仓库内的 ECG-LFM 接口：

```bash
python -m pip install --editable third_party/ECG-LFM
export PYTHONPATH="$PWD:$PWD/third_party/ECG-LFM"
```

原始 ECG、患者级划分、逐记录预测及模型权重不随本仓库发布。训练前需自行准备获准使用的数据与预训练权重，并配置相应路径；详见 [数据说明](DATASET.md)。历史调度脚本中的服务器路径和 GPU 编号需要按实际环境适配。

## Phase 1: Data and Protocol

联合类别为：

| 编号 | 类别 |
|---:|---|
| 0 | OTHER |
| 1 | STEMI + nOMI |
| 2 | STEMI + OMI |
| 3 | NSTEMI + nOMI |
| 4 | NSTEMI + OMI |
| 5 | UA |

归档实验使用 17,960 条 ECG、17,018 名患者，按患者划分外层五折，每个外层训练集再划分互斥的内部训练集和验证集。检查点选择及 OMI 阈值确定仅使用内部验证集。

由六类概率 `q0…q5` 得到 ACS 四状态概率 `[q0, q1+q2, q3+q4, q5]`，OMI 概率为 `q2+q4`。OTHER 表示 non-ACS，不是第四种 ACS 亚型；特殊标签处理见 [标签定义与审计](DATASET.md)。

## Phase 2: Training

数据、划分与预训练权重就绪后，可在已配置环境中启动一个外层折。例如：

```bash
python experiments/ecg_lfm/run_joint6_fold.py \
  --fold 0 --method lora --seed 2026 \
  --output_root results/joint6_reproduction/ecg_lfm \
  --audit_output results/joint6_reproduction/label_audit
```

将 `--method` 改为 `partial` 可运行部分参数微调。五个传统模型的训练入口为 [joint6_traditional_nested.py](experiments/joint6_traditional_nested.py)；完整七模型流程见 [对比调度脚本](scripts/run_joint6_full_comparison_gpu1.sh)。复现时使用新的输出目录，保留原始实验记录。

## Phase 3: Evaluation

对应训练输出的单折评价示例：

```bash
python experiments/ecg_lfm/evaluate_joint6_fold.py \
  --fold 0 --method lora \
  --run_root results/joint6_reproduction/ecg_lfm \
  --output_root results/joint6_reproduction/evaluation \
  --manifest_root manifests/ecg_lfm_nested \
  --device cuda:0 --workers 8 --batch 64
```

报告 Joint-6、ACS、OMI 及 NSTE-ACS 子组指标。所有外层测试均使用已冻结的检查点和阈值；内部验证、外层五折、pooled OOF 与外部测试分别记录。评价实现与回归检查见 [协作说明](CONTRIBUTING.md) 和 [完整评价报告](docs/ECG_LFM_Joint6_Evaluation_Report.md)。

## Results

以下为外层五折的均值 ± 标准差，完整数据见 [汇总表](results/joint6_cv/summary/)。

| 方法 | Joint-6 Macro AUPRC | ACS Macro AUPRC | OMI AUPRC |
|---|---:|---:|---:|
| **ECG-LFM + LoRA** | **0.3843 ± 0.0056** | **0.5445 ± 0.0067** | **0.4299 ± 0.0258** |
| ECG-LFM Partial FT | 0.3738 ± 0.0029 | 0.5322 ± 0.0058 | 0.4103 ± 0.0275 |
| InceptionTime | 0.3567 ± 0.0086 | 0.5108 ± 0.0119 | 0.3860 ± 0.0126 |
| ResNet1D | 0.3506 ± 0.0087 | 0.5044 ± 0.0111 | 0.3742 ± 0.0102 |
| LSTM | 0.3264 ± 0.0077 | 0.4767 ± 0.0100 | 0.3420 ± 0.0264 |
| Transformer1D | 0.3161 ± 0.0134 | 0.4619 ± 0.0133 | 0.3261 ± 0.0310 |
| ViT | 0.3009 ± 0.0106 | 0.4395 ± 0.0119 | 0.3010 ± 0.0267 |

这张表比较完整工程方案：传统模型与 ECG-LFM 的输入预处理不同，不能解释为仅改变网络结构的单变量消融。内部五折结果也不代表独立外部验证或临床部署效果。

## Repository Structure

```text
experiments/   模型训练、评价与统计
configs/       实验及环境配置
scripts/       调度、报告导出与维护入口
tests/         评价和标签处理的回归检查
results/       聚合指标、图表及审计记录
docs/          实验报告、写作交接与日志索引
third_party/   ECG-LFM 等上游实现及来源记录
```

## Documentation

| 内容 | 入口 |
|---|---|
| 主结果与实验边界 | [实验结果总报告](docs/reports/实验结果总报告_最终.md) |
| 论文写作所需证据 | [写作交接说明](docs/PAPER_WRITING_HANDOFF.md) |
| 历史优化与否定结果 | [文档导航](docs/README.md) |
| 结果和日志索引 | [结果目录](results/README.md) · [日志索引](docs/logs/EXPERIMENT_LOG_INDEX.md) |
| 本次整理前的完整首页 | [历史 README](https://github.com/Pendragon08/acs-omi-ami-ecg/blob/2f1761d7561ebea195b54da29b92e4c50a754da5/README.md) |

## Acknowledgements and License

本项目使用 ECG-LFM 及相关开源实现。上游实现、许可和来源保留于 [third_party/](third_party/)，其中 ECG-LFM 许可见 [LICENSE](third_party/ECG-LFM/LICENSE)。

仓库目前未提供统一的顶层 LICENSE；第三方代码与数据继续遵循各自许可。数据使用和发布范围见 [DATASET.md](DATASET.md) 与 [仓库清单](REPOSITORY_MANIFEST.md)。引用本仓库实验时，请注明提交版本、数据划分与评价协议，并引用相应上游方法。
