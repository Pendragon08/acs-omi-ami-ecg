---
experiment_id: ACS-HMOE-WAVELNET-STEMI-NSTEMI-UA-E4-THRESHOLD-YOUDEN-20260826
date: 2026-08-26
status: completed (LEGACY—threshold-only historical task)
task: historical E4 three-independent-sigmoid threshold re-selection
model: Frozen E4 (Xception morphology + LeadSpatial + WavelNet + task routers + KAN heads)
checkpoint_seed: 2026
fold_seed: 2026
threshold_selection: Youden J = sensitivity + specificity - 1 (validation OOF only)
device: NVIDIA RTX A6000 GPU1
---

> **任务定义更正（2026-08-30）**：本文只记录旧三 sigmoid 方案的阈值重选，
> 不能解释为当前 ACS 亚型分类。当前协议为 ACS `OTHER/STEMI/NSTEMI/UA`
> 单标签四分类 + 独立 OMI 二分类；校正后的阈值和结果见
> `corrected_joint_e4_20260829.md`。

# E4 三标签阈值更新：联合最大化 Sensitivity 与 Specificity

## 1. 方法与约束

本次没有重新训练，也没有改变 E4 的网络、预处理、patient-level 五折划分或 checkpoint。使用原 E4 五个冻结模型在各自验证折上的预测，汇总为全体 OOF 验证预测，然后对每个标签独立选择使

`J = Sensitivity + Specificity - 1`

最大的阈值。该目标与最大化 balanced accuracy 等价，体现 Sensitivity 和 Specificity 的联合平衡。隐藏测试集标签没有参与阈值搜索。

原来使用的 OOF F1 最大化阈值为 STEMI=0.28、NSTEMI=0.25、UA=0.24；本次更新后的阈值为：

| 标签 | 原 F1 最大阈值 | 新 Youden/平衡准确率阈值 |
|---|---:|---:|
| STEMI | 0.28 | **0.030540** |
| NSTEMI | 0.25 | **0.035055** |
| UA | 0.24 | **0.323631** |

STEMI 和 NSTEMI 的新阈值较低，原因是 E4 输出概率在这些少数类上偏低；这是阈值重选结果，不代表模型参数发生变化。

## 2. OOF 验证集结果

所有指标均在同一套 17,960 条 OOF 验证记录上计算。AUROC/AUPRC 与阈值无关，Sensitivity、Specificity、PPV、NPV、F1、MCC 和 Accuracy 会随阈值改变。

| 阈值策略 | 标签 | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy | TP/TN/FP/FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Youden | STEMI | 0.9304 | 0.6587 | **0.8613** | **0.8582** | 0.3465 | 0.9861 | 0.4942 | 0.4892 | 0.8585 | 1242/14176/2342/200 |
| Youden | NSTEMI | 0.7931 | 0.2572 | **0.7781** | **0.6775** | 0.1512 | 0.9764 | 0.2532 | 0.2411 | 0.6844 | 961/11331/5394/274 |
| Youden | UA | 0.6534 | 0.4557 | 0.7629 | 0.4678 | 0.4312 | 0.7886 | 0.5510 | 0.2252 | 0.5699 | 4740/5495/6252/1473 |
| **Youden 宏平均** | — | **0.7923** | **0.4572** | **0.8008** | **0.6678** | 0.3097 | 0.9170 | 0.4328 | 0.3185 | 0.7043 | — |

作为对照，旧 F1 阈值在同一 OOF 集上的宏平均为：Sensitivity 0.6250、Specificity 0.7428、PPV 0.4399、F1 0.5027、MCC 0.3630、Accuracy 0.7855。换成 Youden 阈值后，宏平均 Sensitivity 提高 **0.1758**，Specificity 降低 **0.0750**；F1、PPV、MCC 和 Accuracy 下降。这是“提高漏诊召回、牺牲部分阳性预测精度”的明确取舍，并非模型性能全面提升。

## 3. 测试集提交文件

新阈值已固定应用于五个 E4 checkpoint 的测试集概率平均，生成可提交文件；生成过程未读取外部测试标签：

`/data/code/ouyanghao/oyh/ACS-OMI-AMI/results/hmoe_kan_stemi_nstemi_ua_e4_youden/submission_stemi_nstemi_ua_hmoe_e4_youden.csv`

对应服务器结果目录：

`/data/code/ouyanghao/oyh/ACS-OMI-AMI/results/hmoe_kan_stemi_nstemi_ua_e4_youden/`

- `thresholds_youden.json`：阈值和 OOF 指标；
- `e4_oof_test_probabilities.npz`：OOF 标签/概率及测试集概率；
- `oof_predictions.csv`：逐记录 OOF 预测；
- `submission_stemi_nstemi_ua_hmoe_e4_youden.csv`：按 `test.csv` 原始顺序排列的新提交文件。

外部平台尚未对这个新阈值文件返回指标，因此不能把旧 F1 阈值的外部结果当作新阈值结果。提交后应单独记录平台返回的三标签 Sensitivity、Specificity、PPV、NPV、F1 和宏平均值。

## 4. 结论

从现在起，如果目标是“联合最大化 Sensitivity 和 Specificity”，E4 应使用 **Youden 阈值（STEMI 0.030540、NSTEMI 0.035055、UA 0.323631）**。该选择显著提高 STEMI/NSTEMI 的 Sensitivity，但会降低 PPV、F1 和整体 Specificity；它适合优先减少 OMI/AMI 漏诊的工作点，不应描述为 F1 最优方案。若后续需要临床约束（例如 Sensitivity≥0.90），应另行在验证集上定义约束阈值并固定到测试集。
## 5. Recommended operating-point comparison

To avoid selecting a method from one metric alone, a validation-only grid sweep over 0.01–0.99 compared F1 maximization, Youden J maximization, and the highest-specificity points satisfying Sensitivity >= 0.90 or >= 0.95. AUROC and AUPRC are ranking metrics and therefore remain unchanged by threshold selection; the other columns are macro averages over STEMI/NSTEMI/UA.

| Strategy | STEMI/NSTEMI/UA thresholds | AUPRC | AUROC | Sensitivity | Specificity | F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Original F1 maximum | 0.28 / 0.25 / 0.24 | 0.4572 | 0.7923 | 0.6250 | 0.7428 | **0.5027** | **0.7855** |
| Youden J (final) | 0.030540 / 0.035055 / 0.323631 | 0.4572 | 0.7923 | **0.8008** | **0.6678** | 0.4328 | 0.7043 |
| Sensitivity >= 0.90 | 0.01 / 0.01 / 0.22 | 0.4572 | 0.7923 | 0.9355 | 0.3957 | 0.3552 | 0.4899 |
| Sensitivity >= 0.95 | 0.01 / 0.01 / 0.14 | 0.4572 | 0.7923 | 0.9523 | 0.3633 | 0.3524 | 0.4746 |

Without a pre-specified clinical sensitivity floor, **Youden J is the recommended final operating point** because it gives the best validation-set joint Sensitivity/Specificity balance. The >=0.90 and >=0.95 alternatives raise recall but create excessive false positives (NSTEMI specificity 0.1776 at both grid points), so they are retained as safety-oriented reference points rather than the default.
## 6. Model/threshold selection decision

The completed five-fold candidates were ranked primarily by macro AUPRC and secondarily by macro AUROC and the Sensitivity/Specificity operating point. The original E4 five-fold mean was macro AUPRC 0.4689 and macro AUROC 0.7961; the revised sparse HMoE-CWT run was 0.4254 and 0.7698, respectively. The GPU0 historical baseline reported macro AUPRC 0.4765, but it used a different single historical split, so it is not a paired five-fold comparison. On the comparable five-fold evidence, E4 remains the best candidate. Its final operating point is the Youden threshold set above.

## 7. Five-fold breakdown using the final Youden thresholds

The following table applies the same final pooled-OOF thresholds to every validation fold: STEMI `0.030540`, NSTEMI `0.035055`, and UA `0.323631`. Values are macro averages across the three labels within each fold.

| Fold | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8049 | 0.4893 | 0.8147 | 0.6758 | 0.3150 | 0.9163 | 0.4352 | 0.3293 | 0.7027 |
| 1 | 0.7989 | 0.4617 | 0.8109 | 0.6528 | 0.3223 | 0.9262 | 0.4514 | 0.3306 | 0.7092 |
| 2 | 0.7891 | 0.4613 | 0.7878 | 0.6633 | 0.2883 | 0.9097 | 0.4000 | 0.2914 | 0.6832 |
| 3 | 0.7961 | 0.4629 | 0.8308 | 0.6425 | 0.3112 | 0.9290 | 0.4389 | 0.3282 | 0.6927 |
| 4 | 0.7917 | 0.4694 | 0.7597 | 0.7048 | 0.3268 | 0.9113 | 0.4476 | 0.3267 | 0.7335 |
| **Mean ± SD** | **0.7961 ± 0.0062** | **0.4689 ± 0.0119** | **0.8008 ± 0.0276** | **0.6678 ± 0.0241** | **0.3127 ± 0.0149** | **0.9185 ± 0.0087** | **0.4346 ± 0.0204** | **0.3213 ± 0.0167** | **0.7043 ± 0.0191** |

The per-fold/per-label values, including TP/TN/FP/FN, are stored in `fold_metrics_youden.json` and `fold_metrics_youden.csv` under the result directory. These are a descriptive breakdown of the final fixed threshold. Because the threshold was selected from all pooled OOF predictions, this is not a nested threshold-CV estimate.

## 8. Provenance and audit trail

The complete provenance manifest is [e4_youden_provenance_20260826.json](e4_youden_provenance_20260826.json). It records the model variant, loss, seed, patient-level split, GPU, source checkpoint directory, threshold-selection rule, local/remote code paths, output artifacts, and SHA-256 checksums. Historical F1-threshold results remain in the original E4 report; this report contains the final Youden-threshold analysis and its fold breakdown.

## 9. Task-semantics audit

The training CSV contains 17,960 records with mutually exclusive observed states: `000` (no STEMI/NSTEMI/UA) = 9,070, `001` (UA) = 6,213, `010` (NSTEMI) = 1,235, and `100` (STEMI) = 1,442. No training record has two or more positive labels. Therefore:

- at the **model/output level**, the current implementation is a three-label multi-output (three independent sigmoid/one-vs-rest) classifier;
- at the **dataset/task level**, the observed problem is four mutually exclusive states: STEMI, NSTEMI, UA, and Other/negative. It is not a conventional three-class softmax task unless the `000` state is removed or handled separately.

The Youden submission is intentionally evaluated as one-vs-rest multi-label output. It produced 698/1,995 test records with at least two predicted positive labels, illustrating that independent thresholds can violate the observed exclusivity. A clinically exclusive final classifier should therefore be a four-class softmax model, or use a separately pre-specified exclusivity/argmax rule followed by a new evaluation. The current one-vs-rest metrics must not be described as exact-match multiclass accuracy.
