# ACS 亚型与 OMI 联合 ECG 分类

本仓库保存 ACS/OMI ECG 研究中可以公开版本化的代码、配置、聚合实验结果、
脱敏训练日志和中文报告。原始 ECG、患者级表格、checkpoint、逐例预测及提交
CSV 不进入 GitHub。

## 当前任务

当前任务不是四个独立二分类标签，而是一个联合多任务模型：

- **ACS 亚型**：限定 ACS 阳性样本后，在 `STEMI / NSTEMI / UA` 中进行互斥的
  单标签三分类，使用 softmax 和 argmax；
- **OMI**：独立的 OMI/NOMI 二分类，使用 sigmoid，阈值只由内部验证数据选择。

训练全队列明确保留 `[0,0,0]` 的 non-ACS 记录。为此，现有一阶段实现内部使用
`OTHER / STEMI / NSTEMI / UA` 四状态头；`OTHER` 只是 non-ACS 状态，不是第四种
ACS 亚型。报告分别列出全队列状态结果和 ACS 阳性条件三分类结果。

## 最终推荐方案

当前推荐 **ECG-LFM LoRA**（rank=8、alpha=16、dropout=0.05、学习率 `1e-4`）。
患者级五折 pooled OOF 中：

| 任务 | 主要结果 |
|---|---|
| ACS 阳性亚型三分类 | Accuracy `0.8058`，Macro-F1 `0.6541` |
| OMI 二分类（Youden 阈值 0.05） | AUROC `0.9106`，AUPRC `0.4315`，Sensitivity `0.8549`，Specificity `0.8359`，F1 `0.4021` |

外部平台共 1,995 条 ECG，只返回逐类别 hard-label 指标，未返回联合三分类混淆
矩阵或概率，因此不能计算严格的外部三分类总 Accuracy、AUROC 或 AUPRC。

完整结果见 [实验结果总报告_最终.md](docs/reports/实验结果总报告_最终.md)，报告索引见
[docs/reports/README.md](docs/reports/README.md)。

## 当前有效实现

- `experiments/acs_joint_task.py`：标签、患者级五折、指标和阈值公共逻辑；
- `experiments/acs_joint_e4.py`：E4 HMoE/WavelNet/KAN；
- `experiments/acs_joint_model_comparison.py`：五种传统网络对比；
- `experiments/ecg_lfm/run_joint_fold.py`：ECG-LFM 单折训练；
- `experiments/ecg_lfm/evaluate_joint.py`：五折 OOF、图表和外部推理；
- `configs/ecg_lfm/corrected_joint_*.yaml`：Baseline、LoRA、DoRA 配置；
- `scripts/submit_joint_acs_omi.py`：互斥性检查与外部平台提交。

旧的 STEMI/NSTEMI/UA 独立 sigmoid 代码、结果和报告已从当前分支删除，仍可从
Git 历史恢复，但不得用于当前性能结论。

## 结果目录

- `results/corrected_ecg_lfm_joint_bs64_v2/`：ECG-LFM Baseline/LoRA/DoRA；
- `results/corrected_joint_acs_omi_e4/`：E4；
- `results/corrected_joint_model_comparison/`：传统模型；
- `results/loss_feature_ablation/`：历史 OMI/AMI 二分类的损失和特征消融。

每个有效结果目录尽量保存汇总 JSON/CSV、脱敏逐 epoch 日志、损失/LR、ROC/PR、
混淆矩阵和外部平台聚合结果。

## 数据与复现边界

- 训练池：17,960 条 ECG、17,018 位患者；
- ACS 阳性亚型计数：STEMI 1,442、NSTEMI 1,235、UA 6,213；
- non-ACS/OTHER：9,070；OMI 阳性：1,151；
- 患者级五折，seed=2026；外部平台 1,995 条 ECG；
- 训练、验证和外部数据严格隔离；外部标签未在本地读取。

安全与发布规则见 [DATASET.md](DATASET.md) 和
[REPOSITORY_MANIFEST.md](REPOSITORY_MANIFEST.md)。
