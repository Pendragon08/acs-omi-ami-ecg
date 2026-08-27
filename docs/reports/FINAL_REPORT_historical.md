# ACS ECG OMI/AMI 分类 — 最终报告

生成时间：2026-08-17

---

## 0. 摘要（TL;DR）

| 项目 | 最终方案 |
|---|---|
| 模型 | **XceptionTime**（93,121 参数，深度可分离卷积的 InceptionTime 变体） |
| 预处理 | **P3** = 0.5–40Hz 带通滤波 + 全局稳健缩放（median + 1.4826·MAD）+ 在线增强；历史数组的 scaler 在 17,960 例官方训练全集上拟合（含内部 validation） |
| 损失 | **bce**（BCEWithLogitsLoss，无权重） |
| 输出 | 单标签二分类 `[B,1]` → sigmoid（OMI / AMI 分开训练） |
| 超参（OMI） | lr=3e-4，batch=256 |
| 超参（AMI） | lr=1e-4，batch=128 |
| 阈值 | 验证集逐任务 argmax F1（0.05–0.95） |

**官方测试结论**：最终方案 OMI 的 **F1=0.4848 > 原论文基线 0.396**，Acc/Spec/PPV 全面领先，但 Sensitivity（0.500）低于论文基线（0.697）——本文模型更保守（高 Spec 低 Sens）。

---

## 1. 任务与数据

- **任务**：从 12 导联 ECG（100Hz × 10s → `[1000,12]`）分类 OMI（闭塞性心肌梗死）与 AMI（急性心肌梗死），单标签二分类。
- **数据**：官方训练集 17,960 例（含 OMI/AMI 标签，OMI 正类仅 6.4%），官方测试集 1,995 例（标签隐藏）。
- **划分**：按 Patient_id 患者级 90% train / 10% validation（seed=2026，无跨患者泄漏）；官方测试集仅用于最终推理。
- **评估协议（历史实际执行口径）**：
  - 训练/早停/阈值选择使用内部 Validation；Official Test 为独立的 1,995 例隐藏标签测试集。
  - 历史研究过程已多次查看平台反馈，因此本报告中的 Official Test 分数应视为 **development-leaderboard 结果**，不能再宣称为首次盲测确认结果。
  - Validation 主指标 **AUPRC**，优先级链 **AUPRC > AUROC > F1 > Sens/Spec**（不用 Accuracy 选型）。
  - OMI / AMI **分开选型**，不强制同配置。
  - 阈值 = 验证集 argmax F1；最终概率阈值为 OMI 0.48、AMI 0.42，对应验证集 Specificity 约 0.963/0.946。

---

## 2. 方法

### 2.1 预处理 P3（Proposed）

```
unnormalized [1000,12] → 0.5–40Hz 4 阶零相位 Butterworth 带通 (sosfiltfilt, fs=100)
                       → 全局稳健缩放 (median + 1.4826·MAD；历史实现对官方训练全集拟合，包含内部 validation)
                       → 在线增强（仅 train：高斯噪声 p=0.3 / 幅度缩放 p=0.3 / 时间平移 p=0.2）
```
测试集用 P2（= P3 去掉在线增强），套用 train 拟合的 scaler，无再拟合。

### 2.2 模型 XceptionTime

InceptionTime 的深度可分离卷积变体：6 个 Xception 块（128 通道，k=39/19/9 深度可分离 + 1×1 瓶颈 + 池化分支），2 条残差捷径，全局平均池化 → 128 维 → 分类头。仅 93,121 参数。

### 2.3 损失与训练

- **bce**：BCEWithLogitsLoss（无权重），作用于 `[B,1]` raw logits，推理时 sigmoid。
- **训练**：Adam，clip_grad_norm=1.0，epochs=100，早停 patience=15，seed=2026。

---

## 3. 选型过程（Validation，seed=2026）

### 3.1 预处理消融（P0 vs P3）

P3 在 OMI/AMI 上均优于 P0（3 模型综合）：

| 任务 | P0 AUPRC | P3 AUPRC |
|---|---|---|
| OMI | 0.122 | **0.208** |
| AMI | 0.389 | **0.502** |

### 3.2 损失对比（bce / wce / wbce / asl）

bce 在 OMI（AUPRC 0.2275）与 AMI（AUPRC 0.5118）均最优；ASL 是 OMI 上最差。→ **选 bce**。

### 3.3 模型对比（11 模型，bce × P3，验证集 AUPRC 排名）

| 排名 | OMI（AUPRC） | AMI（AUPRC） |
|---|---|---|
| 1 | ResNet1D_Wang 0.3322 | InceptionTime 0.6272 |
| 2 | InceptionTime 0.3184 | ResNet1D_Wang 0.6168 |
| 3 | MyNet6View 0.3179 | MyNet6View 0.6160 |
| 4 | LSTM 0.2744 | LSTM 0.5931 |
| 5 | ViT 0.2631 | ViT 0.5688 |
| 6 | TimesNet 0.2197 | Nonstationary 0.4924 |
| 7 | Informer 0.2194 | Informer 0.4498 |
| 8 | Nonstationary 0.1888 | PatchTST 0.4408 |
| 9 | KANAD 0.1764 | TimesNet 0.4344 |
| 10 | PatchTST 0.1724 | KANAD 0.3487 |
| 11 | MultiPatchFormer 0.1421 | MultiPatchFormer 0.3279 |

结论：轻量 1D 卷积（ResNet1D_Wang / InceptionTime / MyNet6View）在 OMI、AMI 均霸占前三，长时序 Transformer 明显落后。

### 3.4 扩展 CNN 对比 → 选定 XceptionTime

在 ResNet/Inception 家族上做结构性消融（SEResNet1D_Wang / Res2Net1D_Wang / XceptionTime / LITEMV），按「高 Spec 下不塌 Sens」+ 参数量选型：

| 模型 | 参数量 |
|---|---|
| InceptionTime | 473,473 |
| SEResNet1D_Wang | 291,329 |
| ResNet1D_Wang | 286,721 |
| Res2Net1D_Wang | 133,345 |
| **XceptionTime** | **93,121** |
| LITEMV | 85,441 |

**XceptionTime 是唯一在 OMI、AMI 均进前二且不塌 Sens 的模型**，且参数量较小（93,121；LITEMV 更小，为 85,441）→ 选定为最终模型。

### 3.5 超参数调优（lr × batch = 12 run，验证集 argmax-F1 点）

**OMI：**

| lr | batch | AUROC | AUPRC | F1 | Sens | Spec | 早停 epoch |
|---|---|---|---|---|---|---|---|
| 5e-5 | 128 | 0.8829 | 0.3381 | 0.4275 | 0.5957 | 0.9338 | 39 |
| 5e-5 | 256 | 0.8813 | 0.3218 | 0.4490 | 0.5851 | 0.9433 | 39 |
| 1e-4 | 128 | 0.8816 | 0.3396 | 0.4464 | 0.5532 | 0.9486 | 20 |
| 1e-4 | 256 | 0.8811 | 0.3432 | 0.4385 | 0.6064 | 0.9356 | 29 |
| 3e-4 | 128 | 0.8810 | 0.3198 | 0.4292 | 0.5000 | 0.9539 | 7 |
| **3e-4** | **256** | 0.8813 | **0.3493** | **0.4854** | 0.5319 | **0.9634** | 13 |

**AMI：**

| lr | batch | AUROC | AUPRC | F1 | Sens | Spec | 早停 epoch |
|---|---|---|---|---|---|---|---|
| 5e-5 | 128 | 0.8684 | 0.6172 | 0.6173 | 0.6008 | 0.9456 | 23 |
| 5e-5 | 256 | 0.8658 | 0.6250 | 0.6245 | 0.6502 | 0.9320 | 48 |
| **1e-4** | **128** | 0.8760 | **0.6283** | **0.6274** | 0.6132 | **0.9462** | 23 |
| 1e-4 | 256 | 0.8697 | 0.6304 | 0.6211 | 0.6173 | 0.9417 | 23 |
| 3e-4 | 128 | 0.8795 | 0.6384 | 0.6220 | 0.5926 | 0.9507 | 9 |
| 3e-4 | 256 | 0.8776 | 0.6358 | 0.6214 | 0.5844 | 0.9533 | 18 |

**最终超参按「F1 最高」选取**（用户 2026-08-17 指令）：
- **OMI = lr 3e-4 / bs 256**（F1 0.4854，全场最高）
- **AMI = lr 1e-4 / bs 128**（F1 0.6274，全场最高）

> 注：OMI 选 F1 最高（3e-4/256）会让 Sens 从 0.606（1e-4/256）降到 0.532、Spec 顶到 0.963，是「高 Spec 低 Sens」的取舍点，用户已知并确认采用 F1 最高。

---

## 4. 本轮新增实验：focal loss 与多模态融合

为验证「换 focal loss」与「加入 median ECG + 临床特征（gender/age，替换 Time_Interval）」能否进一步提升，做 2×2 对照（均用 3.5 节最终超参，seed=2026）：

### 4.1 全量表（Validation）

| 模型 | Loss | 任务 | AUROC | AUPRC | F1 | Sens | Spec | PPV | NPV | Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| **XceptionTime** | **bce** | **OMI** | **0.8813** | **0.3493** | **0.4854** | 0.5319 | **0.9634** | **0.4464** | 0.9737 | **0.9406** |
| **XceptionTime** | **bce** | **AMI** | 0.8760 | 0.6283 | **0.6274** | 0.6132 | **0.9462** | **0.6422** | 0.9395 | 0.9009 |
| XceptionTime | focal | OMI | **0.8853** | 0.3416 | 0.4370 | **0.6277** | 0.9309 | 0.3352 | **0.9783** | 0.9149 |
| XceptionTime | focal | AMI | 0.8753 | 0.6126 | 0.6000 | 0.6049 | 0.9352 | 0.5951 | 0.9376 | 0.8903 |
| XceptionTimeFusion | bce | OMI | 0.8779 | 0.3193 | 0.4487 | 0.6277 | 0.9350 | 0.3491 | **0.9784** | 0.9188 |
| XceptionTimeFusion | bce | AMI | 0.8739 | 0.6329 | 0.6073 | 0.6173 | 0.9345 | 0.5976 | 0.9394 | 0.8914 |
| XceptionTimeFusion | focal | OMI | 0.8805 | 0.3232 | 0.4000 | 0.4255 | 0.9610 | 0.3774 | 0.9679 | 0.9328 |
| XceptionTimeFusion | focal | AMI | 0.8760 | **0.6418** | 0.6096 | 0.6008 | 0.9417 | 0.6186 | 0.9374 | 0.8953 |

（XceptionTimeFusion = raw 10s 波形 + median beat [100,12] 双编码器 + 临床 MLP，187,425 参数；median beat 由 .med 500Hz→100Hz 重采样 + 带通 + train scaler 得到，按 record_ids 对齐。）

**focal 的 alpha 类别平衡消融**（focal 首次测试用 `alpha=None` 未设类别权重；后按 `alpha = 1 - 正类比例` 补测，即 OMI α=0.9359 / AMI α=0.8508，正类上采样）：

| 模型 | Loss | 任务 | AUROC | AUPRC | F1 | Sens | Spec |
|---|---|---|---|---|---|---|---|
| XceptionTime | focal(α=0.9359) | OMI | 0.8845 | 0.3157 | 0.4218 | 0.6170 | 0.9273 |
| XceptionTime | focal(α=0.8508) | AMI | 0.8732 | 0.6120 | 0.5890 | 0.5309 | 0.9572 |

### 4.2 结论

- **focal 不优于 bce**（对最终 XceptionTime 模型，两任务 AUPRC 均下降：OMI 0.3416 vs 0.3493、AMI 0.6126 vs 0.6283）。
- **focal 即便加类别平衡 alpha 仍不优于 bce**：OMI AUPRC 0.3157 / AMI 0.6120 均低于 bce 的 0.3493 / 0.6283；强上采样正类反而让 OMI 的 AUPRC 进一步下降（Spec 从 0.963 掉到 0.927）。→ **保持 bce（无权重）**。
- **多模态融合不优于纯波形**：OMI 上 AUPRC 0.3193 vs 0.3493（-0.030）、F1 -0.037；AMI 上 AUPRC 仅 +0.005 但 F1 -0.020。→ **保持纯波形输入（不加入 median ECG + 临床特征）**。

**最终方案维持不变：XceptionTime + P3 + bce（纯波形，F1-highest 超参）。**

---

## 5. 最终方案验证集结果

| 任务 | AUROC | AUPRC | F1 | Sens | Spec | PPV | NPV | Acc | 阈值 | 早停 epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| OMI | 0.8813 | 0.3493 | 0.4854 | 0.5319 | 0.9634 | 0.4464 | 0.9737 | 0.9406 | 0.480 | 13 |
| AMI | 0.8760 | 0.6283 | 0.6274 | 0.6132 | 0.9462 | 0.6422 | 0.9395 | 0.9009 | 0.420 | 23 |

---

## 6. 官方测试结果（平台提交，标签隐藏）

| 任务 | Sens | Spec | PPV | NPV | Acc | F1 |
|---|---|---|---|---|---|---|
| OMI | 0.5000 | 0.9614 | 0.4706 | 0.9656 | 0.9318 | **0.4848** |
| AMI | 0.6455 | 0.9393 | 0.6520 | 0.9376 | 0.8952 | **0.6487** |

### 6.1 与原论文 OMI CNN 基线对比

| 方法 | Sens | Spec | PPV | NPV | Acc | F1 |
|---|---|---|---|---|---|---|
| 作者 CNN（原论文） | **0.697** | 0.873 | 0.277 | **0.976** | 0.861 | 0.396 |
| **本方案 XceptionTime** | 0.500 | **0.961** | **0.471** | 0.966 | **0.932** | **0.485** |

判定（按协议）：
- **F1：0.485 > 0.396 ✓**（+0.089，主要综合指标达标）
- **Specificity：0.961 > 0.873 ✓**（大幅更优，误报更少）
- **Accuracy：0.932 > 0.861 ✓** / **PPV：0.471 > 0.277 ✓**
- **Sensitivity：0.500 < 0.697 ✗**（本方案更保守，漏诊更多）
- NPV：0.966 vs 0.976（基本持平）

**解读**：本方案以「高 Spec 低 Sens」的均衡点运行，F1/Acc/Spec/PPV 全面超越论文基线，但灵敏度低于论文 CNN——这是阈值选择（argmax F1 落在高 Spec 区）与 OMI 极度不平衡（6.4% 正类）共同导致的。若临床场景要求高 Sens，可在验证集上向高 Sens 操作点调阈值（如 3.5 节中 1e-4/256 的 Sens 0.606 / Spec 0.936）。

---

## 7. 结论

1. **最终方案**：XceptionTime（93K 参数）+ P3 预处理 + bce 损失，OMI lr=3e-4/bs256、AMI lr=1e-4/bs128，验证集 argmax-F1 阈值。
2. **模型选型**：轻量深度可分离卷积（XceptionTime）在 OMI/AMI 均处第一梯队，参数量远小于 InceptionTime（93K vs 473K）。
3. **损失选型**：bce 稳定最优；focal 不带来增益（本轮 2×2 对照确认）。
4. **输入特征**：纯 12 导联波形已足够；额外加入 median ECG + 临床特征（gender/age）未提升 AUPRC/F1（本轮对照确认），故不采用。
5. **相对论文基线**：本方案 OMI 官方测试 F1 0.485、Spec 0.961、Acc 0.932 高于报告中的原论文 CNN 数值（F1 0.396），但 Sensitivity 较低（0.500 vs 0.697）；该比较仍需补齐原论文结果来源并在新外部队列确认。

---

## 8. 历史方案 GPU1 确定性复跑（2026-08-17）

为纠正此前准备采用新内部划分的方向，本轮恢复并锁定历史数据协议：官方训练集按患者级固定划分为 train 16,174 例 / validation 1,786 例，患者数分别为 15,316 / 1,702，患者交集为 0；外部 Official Test 保持独立，共 1,995 例。划分文件 SHA-256 为 `5bb9ad6f34e8af19a5c7dff1e141c686eec9b6b662edfde997339a4f8cf5dd85`。

在物理 GPU1（NVIDIA RTX A6000）上，从头重跑当前最终方案 XceptionTime + P3 + BCE，seed=2026。除输出目录和结果标签外，模型、数据、初始化、优化器、学习率、batch size、scheduler、epoch、早停、增强与阈值策略均保持历史配置不变。

| 任务 | train 正/负 | val 正/负 | Best epoch | Stop epoch | AUROC | AUPRC | Sens | Spec | PPV | NPV | F1 | Acc | 阈值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OMI | 1,057 / 15,117 | 94 / 1,692 | 13 | 28 | 0.8813 | 0.3493 | 0.5319 | 0.9634 | 0.4464 | 0.9737 | 0.4854 | 0.9406 | 0.48 |
| AMI | 2,436 / 13,738 | 243 / 1,543 | 23 | 38 | 0.8760 | 0.6283 | 0.6132 | 0.9462 | 0.6422 | 0.9395 | 0.6274 | 0.9009 | 0.42 |

一致性核验：

- 新旧 JSON 的完整 validation 指标、训练 history 与最佳 epoch 完全相同。
- 新旧 checkpoint 的 state-dict key 完全一致，全部参数张量逐元素相等，最大绝对差为 0。
- 用新 checkpoint 对 1,995 例 Official Test 重新推理：OMI 预测阳性 136、AMI 预测阳性 296；新旧概率/预测 CSV 的 SHA-256 分别完全相同。
- 因外部预测逐行完全一致，本轮对应的平台指标仍为 OMI F1=0.4848、AMI F1=0.6487；未重复提交平台，避免增加测试集适应性使用。

本轮结论：历史方案已成功恢复且可确定性复现。后续特征和 loss 消融均应固定使用这份患者级 train/validation 划分；超参数和阈值只在 validation 上选择，Official Test 仅在方案冻结后作一次外部评估。为公平比较，历史 P3 scaler 包含 validation 的已知问题在 loss-only 实验中不单独改动，但必须作为方法局限披露；未来严格确认性研究应在患者划分后仅以 fold-train 拟合 scaler。

---

## 附录：复现命令

```bash
# 最终方案训练（OMI / AMI）
python -m exp.exp_acs --model XceptionTime --task omi --loss bce --variant P3 \
  --lr 3e-4 --batch_size 256 --seed 2026 --exp_name final_fusion_focal
python -m exp.exp_acs --model XceptionTime --task ami --loss bce --variant P3 \
  --lr 1e-4 --batch_size 128 --seed 2026 --exp_name final_fusion_focal

# 官方测试推理
python scripts/ecg/predict_official_test.py --model XceptionTime --task omi \
  --loss bce --variant P3 --seed 2026 --exp_name final_fusion_focal
python scripts/ecg/predict_official_test.py --model XceptionTime --task ami \
  --loss bce --variant P3 --seed 2026 --exp_name final_fusion_focal
```
