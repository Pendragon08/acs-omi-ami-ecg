# 校正 ACS 亚型 + OMI 架构对比实验报告（2026-08-29）

> 中文摘要：本实验在完全相同的患者级五折、预处理、优化器和训练预算下，
> 对比 InceptionTime、ResNet1D-Wang、LSTM、ViT-1D、Transformer1D。ACS 采用
> `OTHER/STEMI/NSTEMI/UA` 互斥四分类，OMI 为独立二分类；每个模型的训练日志
> 脱敏 CSV、损失/LR 曲线、ROC/PR、混淆矩阵和外部平台聚合结果已在对应
> `results/corrected_joint_model_comparison/<模型>/` 版本化。中文总览见
> [`实验结果总报告_20260829.md`](实验结果总报告_20260829.md)。

**Status: complete.** Five architectures were rerun under the corrected target
definition on physical GPU1, and each model was submitted once to the external
test platform. This report supersedes the legacy four-independent-sigmoid
comparison; the legacy result is retained only for audit and must not be used
as a final ACS result.

## 1. Correct task and data contract

The source labels are `[STEMI, NSTEMI, UA, OMI]`. The first three are mutually
exclusive ACS subtypes and are encoded as one categorical target:

`OTHER` (`[0,0,0]`), `STEMI`, `NSTEMI`, or `UA`.

OMI is an independent binary target (OMI/NOMI), so `OTHER + OMI` is valid.
Unknown labels are excluded; `[0,0,0]` is retained as an explicit non-ACS
`OTHER` sample. The corrected models emit five logits: four ACS logits and one
OMI logit. ACS is decoded with `argmax`; it is never decoded as three
independent subtype positives.

The development cohort contains 17,960 ECG records from 17,018 patients.
The source audit found 9,070 `OTHER`, 1,442 `STEMI`, 1,235 `NSTEMI`, 6,213
`UA`, and 1,151 OMI-positive records, with no subtype overlap. The hidden
external set contains 1,995 records. The historical patient-level five-fold
split (fold seed 2026) was reused exactly; no new random split was made for
this comparison.

## 2. Fair-comparison protocol

All models used the same fold assignments, fold-fitted robust scaling, the
same train-only augmentation, optimizer and learning-rate budget, batch size
128, learning rate `1e-4`, Adam, no weight decay, ReduceLROnPlateau (factor
0.5, patience 4, minimum learning rate `1e-7`), gradient clipping at 1.0,
maximum 50 epochs, early stopping patience 8, and seed `2026 + fold`.

The loss was identical for all architectures:

`CrossEntropyLoss(ACS categorical) + BCEWithLogitsLoss(OMI binary)`.

No sampler, class reweighting, subtype oversampling, or test-label-derived
parameter was introduced. The OMI threshold was selected from pooled OOF
validation predictions and frozen before external submission. The registered
threshold rule is Youden's index; ACS always uses categorical argmax.

Command (remote execution):

```text
CUDA_VISIBLE_DEVICES=1 python -u experiments/acs_joint_model_comparison.py \
  --models InceptionTime ResNet1D_Wang LSTM ViT Transformer1D \
  --out_dir results/corrected_joint_model_comparison --device cuda:0 \
  --epochs 50 --patience 8 --batch_size 128 --lr 1e-4 --num_workers 0
```

The train/validation curves, ROC/PR curves, confusion matrices, fold histories
and aggregate summaries are in
`results/corrected_joint_model_comparison/<model>/`. The raw screen log is
kept on the execution server at
`logs/corrected_joint_model_comparison/all_gpu1.log`; raw logs, checkpoints,
record-level predictions and ECG data are intentionally not committed.

## 3. Pooled OOF results (registered Youden threshold)

Values below are computed over all five held-out folds. `ACS macro` is the
four-way `OTHER/STEMI/NSTEMI/UA` one-vs-rest macro result. `ACS+` is the
conditional three-way result among ACS-positive records. OMI confusion counts
are `TP/TN/FP/FN` at the model-specific Youden threshold.

| Model | OMI thr | ACS acc | ACS macro-F1 | ACS macro-AUROC | ACS macro-AUPRC | ACS+ acc | ACS+ macro-F1 | OMI AUROC | OMI AUPRC | OMI Sens | OMI Spec | OMI PPV | OMI NPV | OMI F1 | OMI MCC | OMI TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| InceptionTime | 0.06 | 0.5480 | 0.4814 | 0.7638 | 0.5121 | 0.7938 | 0.6280 | 0.8911 | 0.3999 | 0.8358 | 0.8089 | 0.2305 | 0.9863 | 0.3613 | 0.3738 | 962/13597/3212/189 |
| ResNet1D_Wang | 0.05 | 0.5598 | 0.4741 | 0.7648 | 0.5158 | 0.7974 | 0.6187 | 0.8915 | 0.4010 | 0.8384 | 0.8059 | 0.2283 | 0.9865 | 0.3589 | 0.3720 | 965/13547/3262/186 |
| LSTM | 0.06 | 0.5473 | 0.4270 | 0.7459 | 0.4790 | 0.7859 | 0.5954 | 0.8789 | 0.3485 | 0.8158 | 0.8067 | 0.2242 | 0.9846 | 0.3518 | 0.3605 | 939/13560/3249/212 |
| ViT | 0.06 | 0.5311 | 0.4086 | 0.7247 | 0.4430 | 0.7703 | 0.5702 | 0.8560 | 0.3146 | 0.7628 | 0.7927 | 0.2013 | 0.9799 | 0.3185 | 0.3173 | 878/13325/3484/273 |
| Transformer1D | 0.03 | 0.5415 | 0.4596 | 0.7451 | 0.4820 | 0.7813 | 0.6078 | 0.8732 | 0.3623 | 0.7976 | 0.7915 | 0.2076 | 0.9828 | 0.3294 | 0.3349 | 918/13305/3504/233 |

### Five-fold mean ± sample SD

The following means evaluate each fold at the pooled OOF Youden threshold (the
threshold is not reselected on the fold). They are descriptive fold summaries;
the pooled OOF estimate above is the primary registered result.

| Model | ACS acc | ACS macro-F1 | ACS macro-AUROC | ACS macro-AUPRC | ACS+ acc | ACS+ macro-F1 | OMI AUROC | OMI AUPRC | OMI Sens | OMI Spec | OMI F1 | OMI MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| InceptionTime | 0.5480 ± 0.0200 | 0.4631 ± 0.0283 | 0.7727 ± 0.0024 | 0.5266 ± 0.0060 | 0.7938 ± 0.0124 | 0.6259 ± 0.0235 | 0.8964 ± 0.0037 | 0.4103 ± 0.0146 | 0.8358 ± 0.0555 | 0.8090 ± 0.0509 | 0.3676 ± 0.0463 | 0.3794 ± 0.0355 |
| ResNet1D_Wang | 0.5598 ± 0.0081 | 0.4708 ± 0.0295 | 0.7680 ± 0.0029 | 0.5213 ± 0.0088 | 0.7974 ± 0.0075 | 0.6173 ± 0.0187 | 0.8941 ± 0.0036 | 0.4064 ± 0.0093 | 0.8384 ± 0.0388 | 0.8060 ± 0.0375 | 0.3617 ± 0.0281 | 0.3744 ± 0.0221 |
| LSTM | 0.5473 ± 0.0089 | 0.4223 ± 0.0399 | 0.7482 ± 0.0077 | 0.4857 ± 0.0109 | 0.7859 ± 0.0041 | 0.5933 ± 0.0242 | 0.8805 ± 0.0091 | 0.3576 ± 0.0311 | 0.8159 ± 0.0267 | 0.8068 ± 0.0294 | 0.3539 ± 0.0282 | 0.3624 ± 0.0223 |
| ViT | 0.5311 ± 0.0114 | 0.4007 ± 0.0310 | 0.7293 ± 0.0056 | 0.4541 ± 0.0077 | 0.7703 ± 0.0110 | 0.5686 ± 0.0232 | 0.8573 ± 0.0084 | 0.3227 ± 0.0178 | 0.7628 ± 0.0400 | 0.7927 ± 0.0228 | 0.3191 ± 0.0116 | 0.3179 ± 0.0102 |
| Transformer1D | 0.5415 ± 0.0049 | 0.4577 ± 0.0130 | 0.7473 ± 0.0049 | 0.4881 ± 0.0077 | 0.7813 ± 0.0064 | 0.6063 ± 0.0220 | 0.8749 ± 0.0077 | 0.3722 ± 0.0301 | 0.7976 ± 0.0296 | 0.7916 ± 0.0319 | 0.3312 ± 0.0216 | 0.3364 ± 0.0167 |

## 4. Per-class ACS OOF metrics

The table gives AUROC/AUPRC/Sensitivity/Specificity/PPV/NPV/F1/MCC and
`TP/TN/FP/FN` for each one-vs-rest ACS class at the categorical prediction
output. This is not three independent sigmoid classification.

| Model | Class | AUROC | AUPRC | Sens | Spec | PPV | NPV | F1 | MCC | TP/TN/FP/FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| InceptionTime | OTHER | .6786 | .6699 | .7203 | .4957 | .5930 | .6346 | .6505 | .2218 | 6533/4407/4483/2537 |
| InceptionTime | STEMI | .9324 | .6504 | .6158 | .9654 | .6086 | .9664 | .6122 | .5781 | 888/15947/571/554 |
| InceptionTime | NSTEMI | .7985 | .2760 | .2113 | .9740 | .3750 | .9436 | .2703 | .2430 | 261/16290/435/974 |
| InceptionTime | UA | .6456 | .4519 | .3477 | .7762 | .4510 | .6923 | .3927 | .1332 | 2160/9118/2629/4053 |
| ResNet1D_Wang | OTHER | .6780 | .6658 | .7798 | .4403 | .5870 | .6622 | .6698 | .2342 | 7073/3914/4976/1997 |
| ResNet1D_Wang | STEMI | .9333 | .6682 | .5985 | .9720 | .6513 | .9652 | .6238 | .5931 | 863/16056/462/579 |
| ResNet1D_Wang | NSTEMI | .7965 | .2729 | .1595 | .9836 | .4183 | .9406 | .2309 | .2267 | 197/16451/274/1038 |
| ResNet1D_Wang | UA | .6512 | .4565 | .3092 | .8132 | .4668 | .6900 | .3720 | .1386 | 1921/9553/2194/4292 |
| LSTM | OTHER | .6607 | .6466 | .8527 | .3336 | .5663 | .6894 | .6806 | .2183 | 7734/2966/5924/1336 |
| LSTM | STEMI | .9202 | .5937 | .6311 | .9577 | .5659 | .9675 | .5967 | .5604 | 910/15820/698/532 |
| LSTM | NSTEMI | .7743 | .2337 | .1271 | .9831 | .3576 | .9385 | .1876 | .1807 | 157/16443/282/1078 |
| LSTM | UA | .6284 | .4423 | .1656 | .8956 | .4563 | .6699 | .2430 | .0879 | 1029/10521/1226/5184 |
| ViT | OTHER | .6308 | .6093 | .8115 | .3523 | .5611 | .6468 | .6634 | .1845 | 7360/3132/5758/1710 |
| ViT | STEMI | .9011 | .5352 | .5763 | .9510 | .5064 | .9626 | .5391 | .4972 | 831/15708/810/611 |
| ViT | NSTEMI | .7505 | .2011 | .1077 | .9840 | .3325 | .9372 | .1627 | .1573 | 133/16458/267/1102 |
| ViT | UA | .6163 | .4266 | .1954 | .8649 | .4334 | .6702 | .2694 | .0791 | 1214/10160/1587/4999 |
| Transformer1D | OTHER | .6583 | .6432 | .7481 | .4578 | .5847 | .6404 | .6563 | .2153 | 6785/4070/4820/2285 |
| Transformer1D | STEMI | .9197 | .6078 | .6082 | .9606 | .5743 | .9656 | .5908 | .5542 | 877/15868/650/565 |
| Transformer1D | NSTEMI | .7708 | .2366 | .1814 | .9743 | .3425 | .9416 | .2372 | .2103 | 224/16295/430/1011 |
| Transformer1D | UA | .6317 | .4403 | .2962 | .8013 | .4408 | .6828 | .3543 | .1098 | 1840/9413/2334/4373 |

## 5. External hidden-test platform (1,995 records)

The platform receives the corrected schema `ecg_row_record,STEMI,NSTEMI,UA,OMI`.
Each model was submitted once after validation of unique IDs, binary values and
ACS subtype exclusivity. The platform does not expose AUROC/AUPRC; the table
therefore reports its sensitivity, specificity, PPV, NPV, accuracy and F1.

| Model | Label | Sens | Spec | PPV | NPV | Accuracy | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| InceptionTime | OMI | .9062 | .8061 | .2427 | .9921 | .8125 | .3828 |
| InceptionTime | NSTEMI | .1838 | .9796 | .3968 | .9425 | .9253 | .2513 |
| InceptionTime | STEMI | .7593 | .9705 | .6949 | .9785 | .9534 | .7257 |
| InceptionTime | UA | .3152 | .8104 | .4634 | .6950 | .6411 | .3752 |
| ResNet1D_Wang | OMI | .8984 | .8007 | .2361 | .9914 | .8070 | .3740 |
| ResNet1D_Wang | NSTEMI | .1397 | .9866 | .4318 | .9400 | .9288 | .2111 |
| ResNet1D_Wang | STEMI | .6790 | .9793 | .7432 | .9718 | .9549 | .7097 |
| ResNet1D_Wang | UA | .2830 | .8507 | .4961 | .6955 | .6566 | .3604 |
| LSTM | OMI | .8828 | .7922 | .2255 | .9900 | .7980 | .3593 |
| LSTM | NSTEMI | .0735 | .9892 | .3333 | .9359 | .9268 | .1205 |
| LSTM | STEMI | .7160 | .9645 | .6409 | .9746 | .9444 | .6764 |
| LSTM | UA | .1598 | .9337 | .5561 | .6815 | .6692 | .2483 |
| ViT | OMI | .8672 | .7606 | .1989 | .9882 | .7674 | .3236 |
| ViT | NSTEMI | .1103 | .9909 | .4688 | .9384 | .9308 | .1786 |
| ViT | STEMI | .6667 | .9678 | .6467 | .9705 | .9434 | .6565 |
| ViT | UA | .1276 | .9383 | .5179 | .6743 | .6612 | .2047 |
| Transformer1D | OMI | .8828 | .7649 | .2047 | .9896 | .7724 | .3324 |
| Transformer1D | NSTEMI | .1544 | .9785 | .3443 | .9405 | .9223 | .2132 |
| Transformer1D | STEMI | .6543 | .9694 | .6543 | .9694 | .9439 | .6543 |
| Transformer1D | UA | .2654 | .8576 | .4918 | .6921 | .6551 | .3448 |

Platform response/provenance is archived per model in
`results/corrected_joint_model_comparison/<model>/official_platform_4label_metrics.json`
and `external_prediction_provenance_corrected.json`. The submission hashes
are recorded there; hidden test labels were not read locally.

## 6. Curves, checkpoints and audit interpretation

Each model directory contains `training_curves.csv`,
`training_validation_loss_lr.png`, `roc_pr_curves.png`,
`confusion_matrices.png`, five `history_fold*.json` files and an aggregate
`fold_metrics.json`. The architecture script logs train total/ACS-CE/OMI-BCE
loss, validation AUROC/AUPRC/accuracy, joint score and learning rate each
epoch. The comparison curve figure therefore shows the train loss together
with validation metrics and LR; it does not invent a validation loss that the
original comparison loop did not calculate. The best epochs were selected on
the corresponding fold validation score (InceptionTime 14/12/9/23/21,
ResNet1D-Wang 13/15/14/22/16, LSTM 35/43/22/32/22, ViT 6/6/7/8/7,
Transformer1D 16/17/13/15/21).

These are OOF descriptive results rather than a nested estimate: the same
outer-fold validation records select that fold's checkpoint. This limitation
is recorded explicitly and applies equally to every architecture. On the
registered OMI criteria, ResNet1D-Wang has the highest pooled OOF AUROC,
AUPRC, sensitivity and ACS accuracy, while InceptionTime has the highest OOF
F1/MCC and conditional ACS macro-F1; the differences are small. ViT is
consistently weakest. External results preserve the same ordering for OMI F1,
with InceptionTime best among the five (the platform does not provide AUPRC).
The comparison does not establish statistical superiority; paired bootstrap
confidence intervals were not part of this architecture script.

The corrected formulation, not the historical independent-sigmoid output, is
the result eligible for downstream claims or paper tables.
