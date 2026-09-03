# ECG-LFM Nature-style Joint-10 实验完整性审计

**日期：** 2026-09-04

**审计者：** fresh GPT-5.6-Sol ultra，只读、same-family、provisional

**总体结论：** `WARN`
**实验诚信状态：** `WARN（无虚构指标；标签与执行范围仍有限制）`

## A. Ground Truth Provenance：WARN

ACS/OMI 来自真实 `train.csv`、`master_records.csv` 与 `y_all.npy`，审计器按 record、patient 和 label 交叉核对；当前实现进一步验证精确 `[N,4]` 形状、有限二值、AMI 例外、assignment fold/index/patient/ACS/OMI 以及跨折 outer-test 完整覆盖，见 `audit_joint10_labels.py:202-231,372-452`。责任血管仍只是由冠脉分段字段推导的候选标签，不是 clinical-adjudicated canonical culprit ground truth；235 条歧义记录触发阻断，见 `audit_joint10_labels.py:102-157,478-483`。

## B. Score Normalization：PASS

没有指标除以模型自身最大值、均值或预测统计。W0 使用正常 ACS softmax 与独立 OMI sigmoid；OMI threshold 仅在 inner-validation 选择，再冻结到 outer-test，见 `ecg_transformer_classifier.py:86-106` 与 `evaluate_nested_fold.py:183-200`。

## C. Result File Existence：WARN

`audit_summary.json`、`training_gate.json` 和 `baseline_reproduction_status.json` 一致表明 Joint-10/Baseline-Rerun 未运行。主报告将相关指标全部写为 N/A；历史 pooled OOF 与既有 nested Fold0 数字逐项匹配原 JSON，并明确标为背景/非本轮重跑。初审指出 checkpoint 与 Joint-6 取消状态缺少独立产物，现已增加 `checkpoint_inventory.csv` 与 `joint6_cancellation_status.json`。

## D. Dead Code / Control：WARN

本轮只有 pre-training audit，没有 Joint-10 trainer，因此 gate 尚无训练消费者；这是当前 NO-GO 状态下的预期限制。审计器以非零退出码阻断流水线，并写入 `joint10_training_allowed=false`。绘图器现读取 `audit_summary.json`，在图标题显示 `BLOCKED_BY_LABEL_AUDIT`，避免把候选十类分布误读为已完成模型结果。8 项单元测试覆盖映射与 hash；真实 provenance/fold 检查由审计脚本本次运行完成并记录在 `run_manifest.json`。

## E. Scope Assessment：WARN

报告没有声称 Joint-10 有性能提升，也没有把一折结果写成完整 5-fold。`Nature-style` 仅指借鉴 joint-outcome 建模思想；当前项目自定义十类不是论文原始十类的逐类复刻。结论仅为当前公开标签条件下的可行性 NO-GO。

## F. Evaluation Type：PASS

- Joint-10 / Baseline-Rerun：`not_run`，无模型评价；
- Joint-10 产物：`real_gt-derived label feasibility audit`；
- 历史 pooled OOF：`real_gt descriptive non-nested`；
- 既有 nested Fold0 inner-validation：`real_gt model-selection split`；
- 既有 nested Fold0 outer-test：`real_gt held-out single-fold evaluation`。

## Action Items

1. 保持 `BLOCKED_BY_LABEL_AUDIT`，获得可靠单一 culprit artery 后才实现训练器。
2. 未来如新增 Joint-10 runner，启动前必须读取 `training_gate.json` 且仅允许 `joint10_training_allowed=true`。
3. 补充标签后复用当前精确 provenance、AMI、fold 字段和跨折覆盖检查。
4. 不得把自定义 Joint-10 称为 Nature 论文标签空间的精确复现。
5. 审计运行的真实退出码由 `audit_execution_receipt.json` 记录，不以预期退出码代替观察值。

## Claim Impact

- “当前标签不足以可靠训练 Joint-10”：supported。
- “本轮没有生成 Joint-10/Baseline-Rerun 指标”：supported。
- “历史 W0 与既有 nested Fold0 数值”：supported，但只能作为背景。
- “Joint-10 优于/劣于 W0”：unsupported，禁止提出。

## 审计闭环补充

- 审计器已逐项比较 assignment 中的 `patient_id`，并验证全部外层测试患者仅出现一次；患者交叉数为 0，覆盖 17,018 名患者。
- `audit_execution_receipt.json` 记录了审计子进程的真实退出码为 2、状态为 `BLOCKED_BY_LABEL_AUDIT`、`gate_consistent=true`。该退出码是 fail-closed 标签门的预期行为，不是程序崩溃。
- 上述补充关闭了独立复核提出的 provenance 与执行回执缺口；总体仍为 `WARN`，唯一实质限制仍是 235 条 OMI ECG 缺乏可唯一判定的责任大血管标签。
