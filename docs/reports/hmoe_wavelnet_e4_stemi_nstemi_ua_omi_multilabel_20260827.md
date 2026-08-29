> **INVALID FOR CURRENT TASK (2026-08-29)**: this E4 run used four independent sigmoid heads. The observed labels are subtype-exclusive, so its thresholded multi-positive metrics cannot be interpreted as the final ACS/OMI joint task. Corrected rerun: `corrected_joint_e4_20260829.md`.

---
experiment_id: ACS-HMOE-WAVELNET-E4-4LABEL-MULTILABEL-20260827
date: 2026-08-27
status: completed
task: full-cohort four-label multilabel binary classification
labels: STEMI, NSTEMI, UA, OMI
model: E4 (Xception morphology + LeadSpatial GAT + WavelNet + task routers + KAN heads)
loss: BCEWithLogitsLoss (unweighted)
activation: sigmoid
split: patient-level iterative multilabel stratified 5-fold
seed: 2026
fold_seed: 2026
device: NVIDIA RTX A6000 GPU1
---

# E4：STEMI / NSTEMI / UA / OMI 四标签多标签二分类实验

> GitHub 整理版说明：本报告保留实验原始结论和 SHA-256 溯源。受控临床数据、逐例 OOF/测试预测、提交 CSV、患者级 split 和 checkpoint 未纳入默认发布包；本仓库可公开复查的聚合产物位于仓库根目录 results/，详细数据边界见 ../../DATASET.md。

## 1. 结论与最终任务口径

本次完成的是四个独立二分类输出：STEMI、NSTEMI、UA、OMI。每条 ECG 输出四个 logits，并分别经 sigmoid 得到四个概率；训练用未加权 BCEWithLogitsLoss，不是 softmax 三/四分类。

这一口径符合 OMI 可与 ACS 亚型重叠的真实标签结构：训练集中有 STEMI∩OMI=727、NSTEMI∩OMI=423 条 ECG。因此 OMI 不能被塞入与三种 ACS 亚型互斥的单一 softmax 类别。此前讨论的三标签独立输出、四类 softmax、ACS-only 三分类均作为历史实验保留，不与本报告的结果混合比较。

特别按最终要求处理全队列：

- STEMI/NSTEMI/UA=[0,0,0] 保留为明确的 non-ACS-subtype 阴性记录，没有剔除；
- 其中 9,069 条同时为 OMI 阴性（0000），另有 1 条为 OMI 阳性（0001），所以三亚型全零记录共 9,070 条；
- 仅当任一目标标签缺失或不是 0/1 时才会剔除 ECG；本次数据中该数量为 0。

GPU1 五折已经完成，冻结 pooled-OOF Youden J 阈值后已向外部平台提交一份四列预测文件。GPU1 训练完成后已释放。

## 2. 数据审计

训练集包含 17,960 条 ECG、17,018 位患者；外部平台测试集包含 1,995 条 ECG。记录级标签分布如下。

| 标签 | 阳性 | 阴性 | 阳性比例 |
|---|---:|---:|---:|
| STEMI | 1,442 | 16,518 | 8.03% |
| NSTEMI | 1,235 | 16,725 | 6.88% |
| UA | 6,213 | 11,747 | 34.59% |
| OMI | 1,151 | 16,809 | 6.41% |

四标签联合状态（顺序为 STEMI,NSTEMI,UA,OMI）如下，合计恰为 17,960。

| 联合标签 | ECG 数 | 含义 |
|---|---:|---|
| 0000 | 9,069 | non-ACS-subtype、OMI 阴性 |
| 0001 | 1 | non-ACS-subtype、OMI 阳性 |
| 0010 | 6,213 | UA |
| 0100 | 812 | NSTEMI、OMI 阴性 |
| 0101 | 423 | NSTEMI、OMI 阳性 |
| 1000 | 715 | STEMI、OMI 阴性 |
| 1001 | 727 | STEMI、OMI 阳性 |

三种 ACS 亚型在原始标签中不共阳性（STEMI∩NSTEMI=0、STEMI∩UA=0、NSTEMI∩UA=0），但它们在这个多标签任务中仍是三个独立输出头；因此推理时可能产生多亚型同时阳性的模型输出。这是独立二分类的数学行为，不应被误解为训练标签存在三亚型重叠，也不应事后改成 argmax 而改变本任务定义。

机读审计：[label_audit.json](../.remote_project_scan/results/hmoe_kan_stemi_nstemi_ua_omi_e4/label_audit.json)。

## 3. 固定方案与泄漏控制

| 项目 | 设置 |
|---|---|
| 输入 | P2 预处理 ECG，0.5–40 Hz、100 Hz、[1000, 12] |
| 模型 | E4：Xception morphology + LeadSpatial GAT + WavelNet frequency expert + task routers + 4 个 KAN heads |
| 频域实现 | 使用网络内 WavelNet；没有把 MVMD 作为网络外部预处理 |
| 参数量 | 382,446（每折相同） |
| 损失 | BCEWithLogitsLoss()，未加权、无 pos_weight、无过采样/WeightedRandomSampler |
| 激活与决策 | 4 个独立 sigmoid 概率；每个标签使用自己的阈值 |
| 优化 | Adam，lr=1e-4，batch size 128，梯度裁剪 1.0 |
| 训练 | 最多 100 epochs，early stopping patience 15，无 learning-rate scheduler |
| 正则/路由 | gate balance 系数 0.01 |
| 划分 | 患者级 iterative multilabel stratified 5-fold，seed/fold seed 均为 2026 |
| 无泄漏措施 | 同一患者绝不跨 train/validation；每折 robust scaler 只在训练折拟合；数据增强仅在训练折 |
| checkpoint 选择 | 各验证折的 macro-AUROC 最大 |

患者级分层时先对同一患者的所有 ECG 四标签做 OR，再用 MultilabelStratifiedKFold 分折；该做法避免患者泄漏，同时较原先稀有联合状态分层方案更稳健。完整配置：[e4_stemi_nstemi_ua_omi_multilabel.yaml](../configs/e4_stemi_nstemi_ua_omi_multilabel.yaml)，实现：[hmoe_kan_e4_4label_multilabel.py](../.remote_project_scan/experiments/hmoe_kan_e4_4label_multilabel.py)。

执行命令：

    CUDA_VISIBLE_DEVICES=1 python -u experiments/hmoe_kan_e4_4label_multilabel.py \
      --out_dir results/hmoe_kan_stemi_nstemi_ua_omi_e4 \
      --device cuda:0 --seed 2026 --fold_seed 2026 --n_folds 5 \
      --epochs 100 --patience 15 --batch_size 128 --lr 1e-4 \
      --lambda_balance 0.01 --gradient_clip_norm 1.0 --num_workers 4

## 4. 阈值冻结规则

所有阈值仅从五折合并的 OOF 验证概率中，以 0.01–0.99 网格取得；外部测试标签未参与训练、选模或选阈值。最终提交采用兼顾 Sensitivity 与 Specificity 的 Youden J 阈值。

| 策略 | STEMI | NSTEMI | UA | OMI | 用途 |
|---|---:|---:|---:|---:|---|
| Fixed | 0.50 | 0.50 | 0.50 | 0.50 | 参考 |
| Max-F1 | 0.39 | 0.21 | 0.25 | 0.27 | F1 取向的替代运行点 |
| **Youden J（最终）** | **0.06** | **0.04** | **0.35** | **0.03** | **提交平台的固定阈值** |
| Sensitivity ≥ 0.90 | 0.02 | 0.01 | 0.23 | 0.01 | 高敏感度参考 |
| Sensitivity ≥ 0.95 | 0.01 | 0.01 | 0.15 | 0.01* | 高敏感度参考 |

* OMI 在 0.01–0.99 网格内最高 OOF sensitivity 为 0.9279，未达到 0.95；该行显示满足约束失败时的最敏感可得点，不将其伪称为满足 0.95。

评估边界：下文 OOF pooled 是将每例未参与其所属折训练的概率拼接后计算，适合估计排序性能并选择一次最终部署阈值。以这些 pooled-OOF 阈值再回看各验证折的硬分类指标，不是嵌套 CV 的无偏阈值性能估计，故在结果中明确标为描述性逐折稳定性。

## 5. 五折 OOF 验证：最终 Youden J 阈值

### 5.1 合并 OOF 全量指标与混淆矩阵

| 标签 | AUROC | AUPRC | Sens. | Spec. | PPV | NPV | F1 | MCC | Acc. | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STEMI | 0.9313 | 0.6500 | 0.8440 | 0.8772 | 0.3749 | 0.9847 | 0.5192 | 0.5093 | 0.8745 | 1,217 | 14,489 | 2,029 | 225 |
| NSTEMI | 0.7855 | 0.2521 | 0.7798 | 0.6618 | 0.1455 | 0.9760 | 0.2452 | 0.2316 | 0.6699 | 963 | 11,068 | 5,657 | 272 |
| UA | 0.6538 | 0.4638 | 0.7441 | 0.4805 | 0.4310 | 0.7802 | 0.5459 | 0.2179 | 0.5717 | 4,623 | 5,645 | 6,102 | 1,590 |
| OMI | 0.8810 | 0.3744 | 0.8506 | 0.7745 | 0.2052 | 0.9870 | 0.3307 | 0.3466 | 0.7793 | 979 | 13,018 | 3,791 | 172 |
| Macro | 0.8129 | 0.4351 | 0.8046 | 0.6985 | 0.2892 | 0.9320 | 0.4102 | 0.3263 | 0.7239 | — | — | — | — |

这里的 FN 是该标签的漏诊数；OMI 的 pooled OOF FN 为 172/1,151，对应 sensitivity 0.8506。主要瓶颈是 NSTEMI 的低 PPV/AUPRC，以及 UA 的低 specificity；不能仅凭 AUROC 或 Accuracy 判断方案成熟。

### 5.2 五折均值 ± 标准差（同一最终 Youden J 阈值）

| 标签 | AUROC | AUPRC | Sens. | Spec. | PPV | NPV | F1 | MCC | Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STEMI | 0.9376 ± 0.0091 | 0.6731 ± 0.0223 | 0.8439 ± 0.0585 | 0.8771 ± 0.0342 | 0.3852 ± 0.0523 | 0.9849 ± 0.0049 | 0.5242 ± 0.0367 | 0.5149 ± 0.0251 | 0.8744 ± 0.0270 |
| NSTEMI | 0.7945 ± 0.0148 | 0.2651 ± 0.0309 | 0.7791 ± 0.0805 | 0.6615 ± 0.0879 | 0.1499 ± 0.0188 | 0.9767 ± 0.0054 | 0.2496 ± 0.0223 | 0.2360 ± 0.0163 | 0.6697 ± 0.0763 |
| UA | 0.6585 ± 0.0087 | 0.4699 ± 0.0138 | 0.7443 ± 0.0784 | 0.4804 ± 0.0850 | 0.4329 ± 0.0144 | 0.7860 ± 0.0287 | 0.5451 ± 0.0117 | 0.2215 ± 0.0121 | 0.5716 ± 0.0294 |
| OMI | 0.8927 ± 0.0058 | 0.3932 ± 0.0111 | 0.8506 ± 0.0790 | 0.7743 ± 0.0716 | 0.2158 ± 0.0397 | 0.9874 ± 0.0055 | 0.3400 ± 0.0434 | 0.3549 ± 0.0317 | 0.7792 ± 0.0624 |
| Macro | 0.8208 ± 0.0049 | 0.4503 ± 0.0076 | 0.8045 ± 0.0587 | 0.6983 ± 0.0568 | 0.2960 ± 0.0288 | 0.9338 ± 0.0079 | 0.4147 ± 0.0256 | 0.3318 ± 0.0194 | 0.7237 ± 0.0438 |

### 5.3 每折宏平均（最终 Youden J 阈值）

| Fold | 最佳 epoch | AUROC | AUPRC | Sens. | Spec. | PPV | NPV | F1 | MCC | Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 23 | 0.8131 | 0.4428 | 0.7114 | 0.7716 | 0.3408 | 0.9273 | 0.4500 | 0.3545 | 0.7823 |
| 1 | 18 | 0.8190 | 0.4465 | 0.8429 | 0.6664 | 0.2927 | 0.9469 | 0.4230 | 0.3388 | 0.7164 |
| 2 | 14 | 0.8238 | 0.4568 | 0.7603 | 0.7557 | 0.3115 | 0.9247 | 0.4273 | 0.3437 | 0.7607 |
| 3 | 23 | 0.8278 | 0.4618 | 0.8424 | 0.6767 | 0.2786 | 0.9328 | 0.3975 | 0.3236 | 0.7002 |
| 4 | 15 | 0.8204 | 0.4437 | 0.8655 | 0.6212 | 0.2561 | 0.9371 | 0.3759 | 0.2985 | 0.6590 |

## 6. 阈值敏感性（合并 OOF 的宏平均）

AUROC/AUPRC 只由概率排序决定，在这些阈值策略下固定为 0.8129 / 0.4351；下表比较硬分类取舍。

| 阈值策略 | Sens. | Spec. | PPV | NPV | F1 | MCC | Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed 0.50 | 0.3149 | 0.9457 | 0.5248 | 0.8863 | 0.3741 | 0.3170 | 0.8643 |
| Max-F1 | 0.6118 | 0.7862 | 0.4157 | 0.9330 | **0.4857** | **0.3710** | 0.8136 |
| **Youden J（最终）** | **0.8046** | **0.6985** | 0.2892 | 0.9320 | 0.4102 | 0.3263 | 0.7239 |
| Sensitivity ≥ 0.90 | 0.9325 | 0.4568 | 0.2219 | 0.9444 | 0.3400 | 0.2509 | 0.5323 |
| Sensitivity ≥ 0.95 | 0.9533 | 0.4038 | 0.2009 | 0.9406 | 0.3171 | 0.2259 | 0.4949 |

最终使用 Youden J 是为了响应“尽量同时提高 Sensitivity 与 Specificity”的决策目标，而不是因为它使 F1 最大。若临床工作流更重视 PPV/F1，应在新的、预先定义的验证方案中考虑 Max-F1 阈值；不能用外部测试标签反选。

## 7. 外部测试平台结果

提交文件为四个独立 0/1 列：ecg_row_record, STEMI, NSTEMI, UA, OMI。

它由五折 checkpoint 概率平均并应用上节冻结的 Youden 阈值生成。提交前进行了 header、1,995 行数、唯一 ID、取值均为 0/1 的检查；本地没有读取平台测试标签。平台不返回 AUROC 或 AUPRC，因此二者只在本地 OOF 报告。

| 标签 | Sens. | Spec. | PPV | NPV | Acc. | F1 |
|---|---:|---:|---:|---:|---:|---:|
| STEMI | 0.9321 | 0.8576 | 0.3665 | 0.9931 | 0.8637 | 0.5261 |
| NSTEMI | 0.8382 | 0.6003 | 0.1330 | 0.9807 | 0.6165 | 0.2296 |
| UA | 0.7698 | 0.4844 | 0.4368 | 0.8020 | 0.5820 | 0.5573 |
| OMI | 0.9375 | 0.7284 | 0.1914 | 0.9942 | 0.7419 | 0.3179 |
| Macro（简单平均） | 0.8694 | 0.6677 | 0.2819 | 0.9425 | 0.7010 | 0.4077 |

外部测试输出中，STEMI/NSTEMI/UA/OMI 的预测阳性数分别为 412/857/1202/627；无阳性标签 273 条，多标签阳性 820 条。后者在四独立 sigmoid 任务中允许出现，尤其 OMI 与 STEMI/NSTEMI 可共阳性；但对于三个互斥的 ACS 亚型，同时阳性不等于已得到一个可直接用于临床最终分型的互斥结论。

该平台为历史实验期间已经访问过的评测平台。因此这是一份按冻结方案得到的外部平台分数，但不应表述为全新、一次性盲法确认性终点评估；后续正式论文应保留未访问的独立外部队列，或预注册后进行一次最终封存评估。

## 8. 结果解释与下一步判断

- STEMI：排序和外部灵敏度较强（OOF AUROC 0.9313；平台 sensitivity 0.9321），但 Youden 运行点 PPV 仍只有 0.3749/0.3665。
- OMI：OOF AUROC 0.8810、平台 sensitivity 0.9375，适合进一步研究高敏感度筛查，但平台 PPV 0.1914、F1 0.3179，不足以声称可单独确诊。
- NSTEMI：平台 sensitivity 0.8382，但 PPV 0.1330、F1 0.2296，仍是当前方案最明显的薄弱环节。
- UA：F1 相对较高（平台 0.5573），但 specificity 0.4844，假阳性代价需要结合实际流程评估。
- 五折 ranking 指标标准差较小（宏 AUROC 0.8208 ± 0.0049、宏 AUPRC 0.4503 ± 0.0076），但固定阈值下的 sensitivity/specificity 会随折波动，尤其少数类；单次指标不能取代多折结果。

因此，本报告证明了四标签口径、患者级划分和 GPU1 五折管线能够正确运行并获得外部平台分数；它不是与历史三标签/四类任务的配对优越性结论，也尚不支持“NSTEMI 与 UA 已可靠区分”的强临床主张。

## 9. 产物、可追溯性与复查

| 产物 | 路径/说明 |
|---|---|
| 结果目录 | [.remote_project_scan/results/hmoe_kan_stemi_nstemi_ua_omi_e4](../.remote_project_scan/results/hmoe_kan_stemi_nstemi_ua_omi_e4) |
| 五折 checkpoint | e4_4label_fold0_seed2026.pt 至 e4_4label_fold4_seed2026.pt |
| OOF 概率/预测 | [oof_probabilities.npz](../.remote_project_scan/results/hmoe_kan_stemi_nstemi_ua_omi_e4/oof_probabilities.npz)、[oof_predictions.csv](../.remote_project_scan/results/hmoe_kan_stemi_nstemi_ua_omi_e4/oof_predictions.csv) |
| split 与阈值明细 | [split.json](../.remote_project_scan/results/hmoe_kan_stemi_nstemi_ua_omi_e4/split.json)、[threshold_analysis.json](../.remote_project_scan/results/hmoe_kan_stemi_nstemi_ua_omi_e4/threshold_analysis.json) |
| 训练历史 | history_fold0.json 至 history_fold4.json、[run.log](../.remote_project_scan/results/hmoe_kan_stemi_nstemi_ua_omi_e4/run.log) |
| 外部提交 | [submission_stemi_nstemi_ua_omi_e4_youden.csv](../.remote_project_scan/results/hmoe_kan_stemi_nstemi_ua_omi_e4/submission_stemi_nstemi_ua_omi_e4_youden.csv) |
| 平台回包 | [official_platform_4label_metrics.json](../.remote_project_scan/results/hmoe_kan_stemi_nstemi_ua_omi_e4/official_platform_4label_metrics.json)、[official_platform_4label_response.html](../.remote_project_scan/results/hmoe_kan_stemi_nstemi_ua_omi_e4/official_platform_4label_response.html) |
| 完整 provenance | [e4_stemi_nstemi_ua_omi_multilabel_provenance_20260827.json](e4_stemi_nstemi_ua_omi_multilabel_provenance_20260827.json) |

该 provenance 文件包含原始 train/test CSV、代码、配置、split、OOF、阈值、提交文件、平台回包的 SHA-256、运行命令与依赖版本，可用于后续审计和复现。
