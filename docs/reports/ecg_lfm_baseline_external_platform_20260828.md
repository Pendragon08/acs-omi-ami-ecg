# ECG-LFM Baseline 外部平台逐标签盲评记录

- **Experiment ID:** `ECG-LFM-4LABEL-BASELINE-EXTERNAL-20260828`
- **提交时间（服务器 UTC）：** 2026-08-28 10:03
- **任务：** STEMI / NSTEMI / UA / OMI 四个独立 sigmoid 二分类输出（多标签形式）
- **模型：** ECG-LFM full-parameter fine-tuning Baseline
- **集成方式：** 五个历史 patient-level fold 的 record-level 概率算术平均
- **阈值：** 仅由内部 pooled OOF Youden J 确定并锁定：STEMI `0.06`、NSTEMI `0.07`、UA `0.35`、OMI `0.05`
- **外部集：** 1,995 条 ECG；标签由平台隐藏，未在本地读取或用于任何选择
- **平台：** `http://39.105.59.221:8080/`

## 平台返回结果

平台对每个标签分别计算 one-vs-rest 指标。结果按任务标签顺序排列如下：

| Label | Samples | Sensitivity | Specificity | PPV | NPV | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| STEMI | 1,995 | 0.8951 | 0.8843 | 0.4062 | 0.9896 | 0.8852 | 0.5588 |
| NSTEMI | 1,995 | 0.7500 | 0.7127 | 0.1604 | 0.9750 | 0.7153 | 0.2642 |
| UA | 1,995 | 0.7258 | 0.5133 | 0.4365 | 0.7828 | 0.5860 | 0.5452 |
| OMI | 1,995 | 0.8906 | 0.8190 | 0.2522 | 0.9909 | 0.8236 | 0.3931 |
| **逐标签宏平均** | — | **0.8154** | **0.7323** | **0.3138** | **0.9346** | **0.7525** | **0.4403** |

平台接口只返回固定阈值下的硬标签指标，不返回 AUROC、AUPRC、MCC 或混淆矩阵。因此 AUROC/AUPRC 仍应引用内部 pooled OOF 结果，而不能从本次平台响应推算。

## 内部 OOF 对照（非外部测试指标）

Baseline 五折 pooled OOF 的 macro AUROC=`0.8308`、AUPRC=`0.4610`、Sensitivity=`0.7993`、Specificity=`0.7380`、F1=`0.4397`。这些数值与平台外部盲评分开报告。

## 可追溯产物

- 提交 CSV：`results/ecg_lfm_4label/finetune_paper/summary/submission_ecg_lfm_4label_youden.csv`
- 提交 CSV SHA-256：`6467c1f01a9993500d4ffa23e7989f7545c92f2c20d2ac812449000f7a1382d1`
- 平台指标 JSON：`results/ecg_lfm_4label/finetune_paper/summary/official_platform_4label_metrics.json`
- 平台原始响应：`results/ecg_lfm_4label/finetune_paper/summary/official_platform_4label_response.html`
- 提交日志：`results/ecg_lfm_4label/finetune_paper/summary/external_platform_submit.log`

`curl_returncode=0` 且四个标签均返回指标（`missing_metric_labels=[]`），说明本次 Baseline 外部提交成功。
