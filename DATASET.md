# 数据集说明与发布边界

> **任务定义更正（2026-08-29）**：STEMI/NSTEMI/UA 是互斥 ACS 亚型，不能用三个独立 sigmoid 作为最终 ACS 分类器。当前联合任务使用 `OTHER/STEMI/NSTEMI/UA` 四状态单标签 ACS 头（保留 `[0,0,0]` 为 OTHER），并使用独立 OMI/NOMI 二分类头。旧四独立输出实验仅作 legacy 审计，详见 `docs/plans/CORRECTED_TASK_FORMULATION.md`。

## 1. 数据用途

本项目使用 12 导联 ECG 进行 ACS 相关分类研究。当前最终实验使用联合任务：ACS 四状态单标签分类（OTHER/STEMI/NSTEMI/UA）和独立的 OMI/NOMI 二分类。

训练集审计结果为 17,960 条 ECG、17,018 位患者；当前实验使用的外部平台测试集为 1,995 条 ECG。

## 2. 本地数据清单

| 路径 | 文件数/大小 | 作用 |
|---|---:|---|
| med_data/ | 19,955 个 .med，约 228 MB | 原始 ECG 设备格式 |
| ACS-Baseline-main/data/ACS/CSV/train.csv | 约 1.38 MB | 训练元数据、标签与记录关联 |
| ACS-Baseline-main/data/ACS/CSV/test.csv | 约 0.07 MB | 测试元数据与记录关联 |
| ACS-Baseline-main/data/ACS/preprocessed_100hz/ | 39,910 个 .npy，约 1.83 GB | 100 Hz 预处理 ECG |
| data/arrays/（训练服务器） | 本地快照未含 | 最终 E4 使用的 P2 数组、元数据和 scaler |

train.csv 的字段含 Patient_id、ecg_row_record、ecg_med_record、gender、age 及临床标签；即使 Patient_id 是伪标识符，也必须在数据治理方确认其可公开前视为受控数据。

## 3. 当前 GitHub 发行策略

默认仓库不提交以下内容：

- 任何 .med、.npy、.npz 波形和 data/ 目录；
- train.csv、test.csv 和逐例标签/记录关联；
- OOF 概率、逐例预测、测试预测、提交 CSV、split 中的患者/记录标识；
- checkpoint 权重和原始压缩包。

此策略防止临床数据、可链接伪标识符或模型/预测中可能携带的个体信息被意外上传。

## 4. 上传完整数据前必须满足的条件

只有在以下三项均为“是”时，才能将原始或逐例数据推送至 GitHub：

1. 数据拥有者、伦理审批和数据使用协议明确允许向 GitHub（包括 private repository）传输和再分发；
2. 数据已完成可验证的去标识化，并经过负责方审查；仅将 P00001 形式的 ID 视为安全并不充分；
3. 已确认 GitHub LFS 或受控发布资产的存储、带宽、访问权限和长期费用可承担。

本地数据总量约 2.06 GB，且有约 6 万个波形文件。普通 Git 不适合此规模；如获批准，应优先使用 private GitHub repository + Git LFS，或将数据存放在机构受控对象存储中，GitHub 仅提供受控访问申请和校验和。

## 5. 建议的数据交付方式

首选：数据继续保存在机构服务器/获批的对象存储中，GitHub 仓库提供数据访问说明、版本号、样本清单与 SHA-256 校验。

备选：若数据治理方书面批准 GitHub 托管，则在 private 仓库中用 Git LFS 追踪 .med、.npy、.npz 与 CSV；不要提交重复压缩包 ACS-Baseline-main.zip 或 med_data.zip。

## 6. 最终四标签训练所需目录

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

data/arrays 中的 meta 文件需提供 ECG record ID 与 patient ID，以支持患者级五折划分。训练脚本将只剔除四个目标标签中缺失或非 0/1 的 ECG；STEMI/NSTEMI/UA=[0,0,0] 是明确阴性而非缺失。
