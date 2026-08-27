# ACS ECG OMI-AMI 研究代码、方案与实验档案

## 项目概览

本仓库整理了 ACS（急性冠脉综合征）12 导联 ECG 的建模代码、研究方案、实验配置和可复查的聚合结果。项目覆盖历史 AMI/OMI 二分类、STEMI/NSTEMI/UA 任务、损失与特征消融，以及当前最终口径的四标签独立二分类任务。

当前有效任务是对每条 ECG 同时预测 STEMI、NSTEMI、UA 和 OMI 四个二分类标签。模型产生四个独立 logits，使用 sigmoid 转为概率，并在验证集 OOF 概率上为每个标签单独选择阈值。它不是 softmax 的互斥三分类或四分类任务。

当前最终运行保留所有 STEMI/NSTEMI/UA=[0,0,0] 的明确 non-ACS-subtype 记录。OMI 是独立标签，可以与 STEMI 或 NSTEMI 共阳性；因此不能把 OMI 强行塞入与三种亚型互斥的一个类别。

## 当前四标签实验摘要

| 项目 | 设置 |
|---|---|
| 模型 | E4：Xception morphology + LeadSpatial GAT + WavelNet + task routers + 4 个 KAN heads |
| 输出与损失 | 4 个独立 sigmoid 输出；未加权 BCEWithLogitsLoss |
| 训练样本 | 17,960 条 ECG，17,018 位患者 |
| 标签 | STEMI 1,442；NSTEMI 1,235；UA 6,213；OMI 1,151 |
| 划分 | 患者级 iterative multilabel stratified 5-fold，seed=2026 |
| 预处理 | P2 ECG，0.5–40 Hz，100 Hz；每折训练集拟合 robust scaler；增强仅训练折 |
| 训练 | Adam，lr=1e-4，batch size=128，最多 100 epochs，patience=15，无 scheduler |
| 最终阈值 | Youden J：STEMI 0.06，NSTEMI 0.04，UA 0.35，OMI 0.03 |

五折 OOF 宏平均（mean ± SD）为：AUROC 0.8208 ± 0.0049，AUPRC 0.4503 ± 0.0076，Sensitivity 0.8045 ± 0.0587，Specificity 0.6983 ± 0.0568，F1 0.4147 ± 0.0256。

冻结 OOF 阈值后，向外部平台提交的 F1 分别为 STEMI 0.5261、NSTEMI 0.2296、UA 0.5573、OMI 0.3179。详细混淆矩阵、阈值比较、逐折指标、平台回包和 SHA-256 溯源见 docs/reports/hmoe_wavelnet_e4_stemi_nstemi_ua_omi_multilabel_20260827.md。

## 仓库结构

    .
    ├── experiments/          当前 HMoE / WavelNet / KAN 训练脚本
    ├── data_provider/        数据加载与患者级划分代码
    ├── preprocessing/        ECG 预处理、增强、缩放代码
    ├── losses/               BCE、Focal、ASL 等损失实现
    ├── acs_utils/            指标、阈值、随机种子与配置工具
    ├── nets/                 既有时序网络适配器
    ├── scripts/              审计、阈值、消融、预测与提交脚本
    ├── configs/              当前四标签与历史消融配置
    ├── baselines/            原始 ACS 基线模型与训练代码
    ├── docs/
    │   ├── plans/            研究方案、结构设计与改进提示词
    │   └── reports/          可读实验报告及 machine-readable provenance
    ├── results/              不含逐例资料的聚合指标、训练历史和说明
    ├── DATASET.md            数据结构、访问与发布边界
    └── REPOSITORY_MANIFEST.md

## 数据集与访问边界

原始和预处理 ECG 数据不默认纳入 Git 仓库。它们含有 Patient_id、年龄、性别、ECG 记录关联字段及原始/预处理波形，属于受控临床研究数据。

本地完整数据约为 2.06 GB：

| 数据 | 规模 | 说明 |
|---|---:|---|
| 原始 .med ECG | 19,955 文件，约 228 MB | 原始设备格式 |
| 预处理 .npy ECG | 39,910 文件，约 1.83 GB | normalized/unnormalized train/test |
| train.csv / test.csv | 约 1.45 MB | 包含标签和临床关联字段 |

在确认“已脱敏、具备数据使用与再分发授权、且 GitHub 存储方案获批”前，本仓库仅发布数据清单和目录规范，而不会上传逐例波形、CSV、逐例 OOF 预测、外部测试预测或模型权重。详情见 DATASET.md。

## 快速复现

1. 使用获批的本地数据，将最终训练代码需要的数据置于仓库根目录下：

       data/
       ├── arrays/
       │   ├── P2_train.npy
       │   ├── P2_test.npy
       │   ├── meta_train.npz
       │   ├── meta_test.npz
       │   └── scalers.json
       └── CSV/
           ├── train.csv
           └── test.csv

2. 在 Python 环境安装本次运行的核心依赖：PyTorch 2.5.1+cu121、NumPy 2.1.2、scikit-learn 1.7.2、iterative-stratification 0.1.9。HMoE/GAT/KAN 相关依赖以实验脚本的 import 和方案文件为准。

3. 从仓库根目录运行四标签五折训练：

       CUDA_VISIBLE_DEVICES=1 python -u experiments/hmoe_kan_e4_4label_multilabel.py ^
         --out_dir results/hmoe_kan_stemi_nstemi_ua_omi_e4 ^
         --device cuda:0 --seed 2026 --fold_seed 2026 --n_folds 5 ^
         --epochs 100 --patience 15 --batch_size 128 --lr 1e-4 ^
         --lambda_balance 0.01 --gradient_clip_norm 1.0 --num_workers 4

4. 只用验证 OOF 预测选择阈值，再将冻结阈值应用于测试集。不得使用外部测试标签选择阈值或模型。

## 结果解释边界

- 数据集中三种 ACS 亚型的原始标注互斥，但独立多标签头可能同时预测多种亚型；这种输出不能直接等同于临床最终互斥分型。
- OMI 的 OOF AUROC 为 0.8810，外部平台 sensitivity 为 0.9375，但平台 PPV 为 0.1914；目前适合作为高敏感度筛查研究，而非单独确诊工具。
- NSTEMI 的 PPV 与 F1 仍是主要瓶颈。不能只用 Accuracy 或单次运行判断模型优劣。
- 外部平台在历史实验中已访问过；当前分数是冻结方案的外部平台反馈，而不是全新的单次盲法确认性终点。

## 报告与方案索引

| 文档 | 内容 |
|---|---|
| docs/reports/hmoe_wavelnet_e4_stemi_nstemi_ua_omi_multilabel_20260827.md | 当前四标签最终报告：标签审计、五折、阈值、平台结果和解释边界 |
| docs/reports/e4_stemi_nstemi_ua_omi_multilabel_provenance_20260827.json | 当前最终运行的路径、哈希、命令、配置与平台提交溯源 |
| docs/reports/README.md | 全部实验报告索引 |
| docs/plans/ACS_Xception_HMoE_KAN_experiment_plan.md | Xception + HMoE + KAN 实验方案 |
| docs/plans/ACS_STEMI_NSTEMI_UA_Sparse_HMoE_KAN_README.md | Sparse HMoE / KAN 三亚型历史方案 |
| docs/plans/ACS_ECG_OMI_AMI_multilabel_master_plan.md | OMI/AMI 多标签总体研究方案 |

## 发布状态

本仓库准备为 private GitHub research repository。代码、配置、方案和不含逐例信息的实验报告可安全同步；原始数据上传与仓库是否公开，须在 DATASET.md 所列数据治理条件确认后再进行。

