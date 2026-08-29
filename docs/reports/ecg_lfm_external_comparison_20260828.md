# 四标签外部平台逐标签对比：ECG-LFM Baseline vs E4

本报告比较同一外部盲评集（1,995 条 ECG）上已经提交并成功返回平台指标的两个四标签模型：

- **ECG-LFM Baseline**：五折 record-level 概率平均，OOF Youden 阈值 `STEMI=0.06, NSTEMI=0.07, UA=0.35, OMI=0.05`。
- **Historical E4**：HMoE/WavelNet/KAN 四标签方案，已保存的提交阈值为 `STEMI=0.06, NSTEMI=0.04, UA=0.35, OMI=0.03`。

两者均为四个独立 sigmoid 二分类输出；平台按 one-vs-rest 返回硬标签指标。平台不返回 AUROC/AUPRC/MCC，因此这些排序指标不能从外部盲评表推算。

## 外部平台逐标签结果

| Model | Label | Sensitivity | Specificity | PPV | NPV | Accuracy | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ECG-LFM Baseline | STEMI | 0.8951 | 0.8843 | 0.4062 | 0.9896 | 0.8852 | 0.5588 |
| Historical E4 | STEMI | 0.9321 | 0.8576 | 0.3665 | 0.9931 | 0.8637 | 0.5261 |
| ECG-LFM Baseline | NSTEMI | 0.7500 | 0.7127 | 0.1604 | 0.9750 | 0.7153 | 0.2642 |
| Historical E4 | NSTEMI | 0.8382 | 0.6003 | 0.1330 | 0.9807 | 0.6165 | 0.2296 |
| ECG-LFM Baseline | UA | 0.7258 | 0.5133 | 0.4365 | 0.7828 | 0.5860 | 0.5452 |
| Historical E4 | UA | 0.7698 | 0.4844 | 0.4368 | 0.8020 | 0.5820 | 0.5573 |
| ECG-LFM Baseline | OMI | 0.8906 | 0.8190 | 0.2522 | 0.9909 | 0.8236 | 0.3931 |
| Historical E4 | OMI | 0.9375 | 0.7284 | 0.1914 | 0.9942 | 0.7419 | 0.3179 |

逐标签算术宏平均：Baseline `Sensitivity=0.8154, Specificity=0.7323, PPV=0.3138, NPV=0.9346, Accuracy=0.7525, F1=0.4403`；E4 `Sensitivity=0.8694, Specificity=0.6677, PPV=0.2819, NPV=0.9425, Accuracy=0.7010, F1=0.4077`。

## Baseline − E4 差值

| Label | ΔSensitivity | ΔSpecificity | ΔPPV | ΔNPV | ΔAccuracy | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|
| STEMI | -0.0370 | +0.0267 | +0.0397 | -0.0035 | +0.0215 | +0.0327 |
| NSTEMI | -0.0882 | +0.1124 | +0.0274 | -0.0057 | +0.0988 | +0.0346 |
| UA | -0.0440 | +0.0289 | -0.0003 | -0.0192 | +0.0040 | -0.0121 |
| OMI | -0.0469 | +0.0906 | +0.0608 | -0.0033 | +0.0817 | +0.0752 |
| 宏平均 | -0.0540 | +0.0646 | +0.0319 | -0.0079 | +0.0515 | +0.0326 |

## 解读

- E4 的四个标签 Sensitivity 整体更高，尤其 NSTEMI/OMI；但 Specificity 明显下降，PPV 和 Accuracy 也更低。
- Baseline 的外部宏平均 F1、Specificity、PPV 和 Accuracy 更高，说明其工作点更保守、假阳性更少。
- UA 是例外：E4 F1 `0.5573` 略高于 Baseline `0.5452`，但 Specificity 仍低于 Baseline。
- 因此不能只按 Sensitivity 判定 E4 更好；若目标是减少漏诊可偏向 E4，若要求 Sensitivity/Specificity 平衡和较少误报，Baseline 更合适。

## 传统对比模型的外部结果状态

五个传统四标签模型已经使用各自五折 checkpoint、各自 pooled-OOF Youden 阈值和同一 1,995 条外部 ECG 集完成平台盲评。平台返回的逐标签结果如下。

### STEMI

| Model | Sensitivity | Specificity | PPV | NPV | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ECG-LFM Baseline | 0.8951 | 0.8843 | 0.4062 | 0.9896 | 0.8852 | 0.5588 |
| Historical E4 | 0.9321 | 0.8576 | 0.3665 | 0.9931 | 0.8637 | 0.5261 |
| InceptionTime | 0.9444 | 0.8402 | 0.3430 | 0.9942 | 0.8486 | 0.5033 |
| ResNet1D_Wang | 0.9259 | 0.8685 | 0.3836 | 0.9925 | 0.8732 | 0.5425 |
| LSTM | 0.9012 | 0.8582 | 0.3596 | 0.9899 | 0.8617 | 0.5141 |
| Transformer1D | 0.9444 | 0.7780 | 0.2732 | 0.9937 | 0.7915 | 0.4238 |
| ViT | 0.9012 | 0.7054 | 0.2128 | 0.9878 | 0.7213 | 0.3443 |

### NSTEMI

| Model | Sensitivity | Specificity | PPV | NPV | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ECG-LFM Baseline | 0.7500 | 0.7127 | 0.1604 | 0.9750 | 0.7153 | 0.2642 |
| Historical E4 | 0.8382 | 0.6003 | 0.1330 | 0.9807 | 0.6165 | 0.2296 |
| InceptionTime | 0.7868 | 0.6498 | 0.1412 | 0.9766 | 0.6591 | 0.2394 |
| ResNet1D_Wang | 0.7574 | 0.6982 | 0.1551 | 0.9752 | 0.7023 | 0.2575 |
| LSTM | 0.7941 | 0.6364 | 0.1378 | 0.9769 | 0.6471 | 0.2348 |
| Transformer1D | 0.7941 | 0.6278 | 0.1350 | 0.9766 | 0.6391 | 0.2308 |
| ViT | 0.8603 | 0.3954 | 0.0943 | 0.9748 | 0.4271 | 0.1699 |

### UA

| Model | Sensitivity | Specificity | PPV | NPV | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ECG-LFM Baseline | 0.7258 | 0.5133 | 0.4365 | 0.7828 | 0.5860 | 0.5452 |
| Historical E4 | 0.7698 | 0.4844 | 0.4368 | 0.8020 | 0.5820 | 0.5573 |
| InceptionTime | 0.7757 | 0.4501 | 0.4229 | 0.7944 | 0.5614 | 0.5473 |
| ResNet1D_Wang | 0.8343 | 0.3938 | 0.4168 | 0.8206 | 0.5444 | 0.5559 |
| LSTM | 0.6950 | 0.5240 | 0.4313 | 0.7679 | 0.5825 | 0.5323 |
| Transformer1D | 0.8695 | 0.3420 | 0.4070 | 0.8346 | 0.5223 | 0.5545 |
| ViT | 0.5367 | 0.5682 | 0.3923 | 0.7024 | 0.5574 | 0.4533 |

### OMI

| Model | Sensitivity | Specificity | PPV | NPV | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ECG-LFM Baseline | 0.8906 | 0.8190 | 0.2522 | 0.9909 | 0.8236 | 0.3931 |
| Historical E4 | 0.9375 | 0.7284 | 0.1914 | 0.9942 | 0.7419 | 0.3179 |
| InceptionTime | 0.8828 | 0.8082 | 0.2399 | 0.9902 | 0.8130 | 0.3773 |
| ResNet1D_Wang | 0.9062 | 0.7933 | 0.2311 | 0.9920 | 0.8005 | 0.3683 |
| LSTM | 0.9375 | 0.7242 | 0.1890 | 0.9941 | 0.7378 | 0.3145 |
| Transformer1D | 0.9141 | 0.7413 | 0.1950 | 0.9921 | 0.7524 | 0.3214 |
| ViT | 0.9062 | 0.6690 | 0.1580 | 0.9905 | 0.6842 | 0.2691 |

### 四标签逐标签宏平均

| Model | Sensitivity | Specificity | PPV | NPV | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ECG-LFM Baseline | 0.8154 | 0.7323 | 0.3138 | 0.9346 | 0.7525 | 0.4403 |
| ResNet1D_Wang | 0.8559 | 0.6885 | 0.2966 | 0.9451 | 0.7301 | 0.4310 |
| InceptionTime | 0.8474 | 0.6871 | 0.2868 | 0.9388 | 0.7205 | 0.4168 |
| Historical E4 | 0.8694 | 0.6677 | 0.2819 | 0.9425 | 0.7010 | 0.4077 |
| LSTM | 0.8319 | 0.6857 | 0.2794 | 0.9322 | 0.7073 | 0.3989 |
| Transformer1D | 0.8805 | 0.6223 | 0.2525 | 0.9493 | 0.6763 | 0.3826 |
| ViT | 0.8011 | 0.5845 | 0.2143 | 0.9139 | 0.5975 | 0.3091 |

各传统模型的四标签平台请求均 `curl_returncode=0` 且返回四个标签。与 Baseline 相比，传统模型普遍通过降低 Specificity 换取 Sensitivity；Baseline 的外部宏平均 Specificity、PPV、Accuracy 和 F1 均最高。

## 证据文件

- Baseline 指标：`results/ecg_lfm_4label/finetune_paper/summary/official_platform_4label_metrics.json`
- Baseline 原始响应：`results/ecg_lfm_4label/finetune_paper/summary/official_platform_4label_response.html`
- E4 指标：`.remote_project_scan/results/hmoe_kan_stemi_nstemi_ua_omi_e4/official_platform_4label_metrics.json`
- 四标签传统模型外部指标：`results/model_comparison_4label/<model>/official_platform_4label_metrics.json`
- 四标签传统模型平台原始响应：`results/model_comparison_4label/<model>/official_platform_4label_response.html`
- 四标签传统模型提交文件：`results/model_comparison_4label/<model>/submission_4label_<model>.csv`
- 四标签传统模型内部 OOF 对比：`reports/model_comparison_4label_20260828.md`
