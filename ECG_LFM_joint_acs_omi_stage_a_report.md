# ECG-LFM ACS×OMI Joint-7 Stage-A 报告

## 1. Repository audit

当前 W0 真实结构为共享 ECG-LFM+LoRA 后的单个 `Linear(1024,5)`，前四个 logits 使用 ACS softmax+CE，第五个 logit 使用独立 OMI sigmoid+BCE。它在语义上是 factorized two-head，但并非两个物理 Linear 层。旧实现和 checkpoint 保留不变；Joint-7 已作为独立 formulation、criterion、配置和执行路径实现。

| 模块 | 真实路径 | 关键实现 | 本次处理 |
|---|---|---|---|
| Dataset / dataloader | `fairseq_signals/tasks/ecg_classification.py`、`data/ecg/raw_ecg_dataset.py` | `ECGDiagnosisTask`、`FileECGDataset` | 保留 manifest 和 `.mat idx` 读取；新增 formulation 配置 |
| W0 标签与 Loss | `experiments/ecg_lfm/run_joint_fold.py`、`criterions/acs_joint.py` | `ensure_joint_label_file`、`ACSJointCriterion` | 原样保留 |
| Nested patient split | `experiments/ecg_lfm/build_nested_manifests.py` | `StratifiedGroupKFold` | 不重划分，复用原 hash |
| ECG-LFM / LoRA / pooled feature | `models/classification/ecg_transformer_classifier.py`、`modules/parameter_efficient.py` | `ECGTransformerClassificationModel`、`LoRALinear` | Backbone 不改；Joint-7 输出 7 logits |
| Joint-7 标签 | `experiments/ecg_lfm/joint7.py`、`audit_joint7_labels.py` | `encode_source_labels`、`one_hot_joint7` | 新增映射、异常处理与数据硬门 |
| Joint-7 Loss | `criterions/acs_joint7.py` | `ACSJoint7Criterion` | 只计算 unweighted Joint CE |
| 训练入口 / 配置 | `run_joint7_fold.py`、`corrected_joint7_lora.yaml` | Fold runner / Hydra config | 保持 W0 超参数，训练前强制审计 |
| 评估与校准 | `evaluate_joint7_stage_a.py` | 边缘化、指标、temperature scaling、GO gate | 仅 inner-validation；门控前不加载 outer/external |

## 2. Joint-label audit

审计结论：**PRE-TRAINING DATA GATE STOP**。

停止原因：major Joint-7 class has N=0 in inner-train: UA+OMI。

`OTHER+OMI=1`，符合 singleton 异常规则：其 joint target 为 OTHER，原始 OMI=1 不改，primary OMI 评价仍应包含它。

### 原始八组合

| 原始组合 | ECG | 患者 | prevalence |
|---|---:|---:|---:|
| OTHER+nOMI | 9069 | 8935 | 0.504955 |
| OTHER+OMI | 1 | 1 | 0.000056 |
| STEMI+nOMI | 715 | 713 | 0.039811 |
| STEMI+OMI | 727 | 725 | 0.040479 |
| NSTEMI+nOMI | 812 | 803 | 0.045212 |
| NSTEMI+OMI | 423 | 417 | 0.023552 |
| UA+nOMI | 6213 | 5812 | 0.345935 |
| UA+OMI | 0 | 0 | 0.000000 |

### Nested Fold0 七联合类

| subset | joint class | ECG | 患者 | prevalence |
|---|---|---:|---:|---:|
| inner_train | OTHER | 5816 | 5729 | 0.506047 |
| inner_train | STEMI+nOMI | 456 | 454 | 0.039676 |
| inner_train | STEMI+OMI | 454 | 453 | 0.039502 |
| inner_train | NSTEMI+nOMI | 535 | 529 | 0.046550 |
| inner_train | NSTEMI+OMI | 272 | 268 | 0.023667 |
| inner_train | UA+nOMI | 3960 | 3703 | 0.344558 |
| inner_train | UA+OMI | 0 | 0 | 0.000000 |
| inner_validation | OTHER | 1430 | 1408 | 0.499825 |
| inner_validation | STEMI+nOMI | 119 | 119 | 0.041594 |
| inner_validation | STEMI+OMI | 125 | 125 | 0.043691 |
| inner_validation | NSTEMI+nOMI | 115 | 113 | 0.040196 |
| inner_validation | NSTEMI+OMI | 68 | 67 | 0.023768 |
| inner_validation | UA+nOMI | 1004 | 946 | 0.350926 |
| inner_validation | UA+OMI | 0 | 0 | 0.000000 |
| outer_test | OTHER | 1824 | 1799 | 0.505824 |
| outer_test | STEMI+nOMI | 140 | 140 | 0.038824 |
| outer_test | STEMI+OMI | 148 | 147 | 0.041043 |
| outer_test | NSTEMI+nOMI | 162 | 161 | 0.044925 |
| outer_test | NSTEMI+OMI | 83 | 82 | 0.023017 |
| outer_test | UA+nOMI | 1249 | 1163 | 0.346367 |
| outer_test | UA+OMI | 0 | 0 | 0.000000 |

## 3. Provenance

- y_all 与 master index/record label 逐条不一致：0
- 检查的预处理 segment `.mat` 数：35920
- `.mat` 内嵌 idx 与 segment/master idx 不一致：0
- patient-level split：沿用既有 nested Fold0，未重划分
- outer-test：只做标签支持度审计，没有模型推理、调参或指标计算
- external-test：未访问

## 4. Joint-7 implementation status

已实现固定 7-class mapping、OTHER+OMI singleton 策略、Joint CE、Joint softmax 的 ACS/OMI/NSTE 概率边缘化、Joint-7 内部验证 evaluator、temperature scaling、fixed-sensitivity operating points、完整预测字段、图表和 GO gate。W0 路径继续可运行。

验证结果：Joint-7 新增 12 项单元测试和原 W0 评估 6 项回归测试全部通过；Hydra/Fairseq 对 Joint-7 配置的 schema compose 通过；既有 W0 Fold0 checkpoint 已在修改后重新只读加载成功，确认 `output_semantics=joint`、`proj=(5,1024)`、LoRA 路径不变。记录保存在 `docs/logs/joint7_unit_tests.txt`、`joint7_config_compose.txt` 和 `joint7_w0_checkpoint_compatibility.txt`。

## 5. Training configuration

Joint-7 配置保持 W0 的 pretrained ECG-LFM、LoRA r=8/alpha=16/dropout=0.05、Adam、LR=1e-4、batch=64、累积4、Step scheduler、100 epochs、patience=10；checkpoint criterion 为 inner-validation joint CE 最小。

## 6. Stage-A results

未启动训练，因此没有 J1 指标，也没有人为填充或虚构 `UA+OMI` 指标。`UA+OMI` 全局 support=0，CE 训练中永远没有该类正 target，AUROC/AUPRC/F1/Sensitivity 无法有效估计。

## 7. GO / CONDITIONAL GO / NO-GO

**NO-GO（数据可行性硬门）**。

依据所给完整方案：任一主要 joint class 在 inner-train 中 N=0 必须立即停止。Fold0 的 `UA+OMI` 满足该停止条件。此处 NO-GO 判断针对预注册的固定 Joint-7 标签空间，并不代表所有联合/层级建模思想均无效。

## 8. 是否建议完整 nested 5-fold

不建议，也未启动。需要先由研究者重新确认科学标签空间，例如是否有可靠 UA+OMI 病例、是否应采用与现有数据支持一致的联合/层级 formulation；本轮不擅自更改标签、不删除样本、不新增 loss 或其他变量。

## 9. 可追溯输出

聚合审计位于 `results/joint7_stage_a/joint_label_audit/`。`exceptions.csv` 含 record/patient 标识，只保留在受控服务器，不提交 Git。
