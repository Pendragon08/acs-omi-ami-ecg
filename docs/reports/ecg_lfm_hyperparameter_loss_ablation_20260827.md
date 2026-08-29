# ECG-LFM 四标签超参数与类别不平衡损失实验报告

更新时间：2026-08-28（Asia/Shanghai）  
状态：**Baseline Fold 0–2 已完成并补齐 record-level 正式评价，Fold 3 正在 GPU0 训练（当前到 epoch 8；最佳 validation loss 暂为 epoch 6 的 0.406），Fold 4 尚未开始；调参队列在 GPU1 已完成 5 个候选、1 个运行中、1 个排队。完整五折 OOF、与 E4 的患者级 paired bootstrap、唯一最终方案冻结和外部平台提交尚未完成。**  
> 这是 2026-08-28 最终 LoRA/DoRA 运行完成前的阶段性快照；最终适配策略与完整结果以 [`ecg_lfm_local_best_lora_dora_optimization_20260828.md`](ecg_lfm_local_best_lora_dora_optimization_20260828.md) 为准。
审稿式消融状态：**pending external review**（当前会话禁止启动独立评审代理，先执行预注册高优先级实验）。

## 1. 目标和数据集

任务为四个独立 sigmoid 二分类输出，固定标签顺序为 `STEMI,NSTEMI,UA,OMI`，不是互斥 Softmax 四分类。明确的 non-ACS `[0,0,0]` 样本作为负样本保留。

- 内部队列：17,960 ECG / 17,018 patients。
- 外部盲测队列：1,995 ECG。
- 全队列：19,955 ECG。
- 内部阳性数：STEMI 1,442；NSTEMI 1,235；UA 6,213；OMI 1,151。
- 切分：精确复用历史 E4 patient-level iterative multilabel 5-fold，seed 2026。
- 外部队列不参与 loss 权重、学习率、batch、checkpoint、阈值或方案选择。

用户提出“根据验证和测试指标调参”。为避免 test leakage，本实验将其落实为：**所有调参只根据内部 patient-level validation/OOF 指标；外部平台仅在最终方案和 OOF 阈值锁定后提交一次。外部结果只用于最终泛化评价，不反向调参。**

## 2. 固定不变的部分

除正在比较的学习率、有效 batch 或 loss 外，其余均固定：

- 官方 ECG-LFM commit `647369a012dd8d5d362e99e2bf238c91f3e1d03b`。
- 官方预训练权重 MD5 `afa8dfc680590948f82aabbdb123bf43`。
- 原始 500 Hz、12 导联、10 秒 ECG；每条记录分成两个不重叠 5 秒 segment。
- 1 Hz high-pass、50/60 Hz notch、30 Hz low-pass、逐记录逐导联 z-score。
- 两个 segment logits 先求均值，再 sigmoid 得到 record-level probability。
- 官方 1024→4 线性分类头、Adam `(0.9,0.98)`、`eps=1e-8`。
- 最多 100 epochs，validation-loss strict-best checkpoint，early stopping patience 10。
- seed 2026、相同初始化策略、相同数据顺序、无 sampler、无 oversampling、无数据增强。
- 每个方案独立用 record-level OOF 概率在 `0.01–0.99` 搜索逐标签 Youden J 阈值。

## 3. Baseline

Baseline 是当前 `finetune_paper`：

| 参数 | 值 |
|---|---|
| Loss | unweighted `BCEWithLogitsLoss` |
| LR | `1e-6` |
| Physical batch | 32 |
| Gradient accumulation | 8 |
| Effective batch | 256 |
| Scheduler | step approximation of epoch-wise exponential decay，factor `0.99` |
| Checkpoint | minimum validation loss |

当前训练器中间轨迹（不能当作最终 record-level 结果）：epoch 1–5 validation loss `0.452→0.412`，AUROC `0.8325→0.8697`，AUPRC `0.4135→0.4886`。到 epoch 10，train loss 已继续下降到 `0.339`，但 validation loss 升至 `0.434`、AUPRC 回落至 `0.4830`，strict-best checkpoint 仍为 epoch 5。固定阈值 `F1@0.5` 同期反而升到 `0.4621`，说明阈值穿越/校准变化不能替代 validation loss checkpoint 与 OOF threshold 评价。

![Baseline Fold 0 training curves](../artifacts/ecg_lfm/live_finetune_paper_fold0_training_validation_curves.png)

## 4. Weighted BCE 实现与权重

使用官方 criterion 已支持的 numerically stable：

```text
BCEWithLogitsLoss(pos_weight=N_negative/N_positive)
```

四个权重按**每个 fold 的唯一训练记录**单独计算；validation 和 external 完全排除。没有同时使用 sampler/oversampling，也不重复叠加其他 class weight。官方 criterion 与 PyTorch 参考实现的合成输入绝对差为 `0.0`。

| Fold | Train records | STEMI | NSTEMI | UA | OMI |
|---:|---:|---:|---:|---:|---:|
| 0 | 14,354 | 11.4385 | 13.4990 | 1.8916 | 14.6022 |
| 1 | 14,378 | 11.4701 | 13.5674 | 1.8889 | 14.6283 |
| 2 | 14,364 | 11.4471 | 13.4945 | 1.8878 | 14.5792 |
| 3 | 14,357 | 11.4411 | 13.5167 | 1.8963 | 14.5716 |
| 4 | 14,387 | 11.4779 | 13.6358 | 1.8890 | 14.6380 |

权重取证：`artifacts/ecg_lfm/contract_tests_weighted_bce.log`。每次实际训练的正负数、比例和权重还会写入该 fold 的 `launch_provenance.json`。

Fold 0 Weighted BCE 单个 optimizer update 真机门禁已在 GPU1 通过：官方 Hydra 配置实际加载上述四个权重，无 OOM/NaN；train loss=`2.070`、valid loss=`2.174`、gradient norm=`8.053`、训练步后剩余显存约 28.2 GiB。加权 loss 的绝对数值不能与 unweighted BCE loss 横向比较；较大的初始 gradient norm 也是优先测试 `3e-7` 加权候选的原因。

## 5. 预注册消融矩阵

完整机器可读计划：`configs/ecg_lfm/tuning_experiments.json`。

| Run | LR | Effective batch | Loss | 具体问题 | 状态 |
|---|---:|---:|---|---|---|
| `finetune_paper` | 1e-6 | 256 | BCE | baseline | Fold 0–2 complete；Fold 3 running / GPU0；Fold 4 pending |
| `lr3e7_bce_b256` | 3e-7 | 256 | BCE | 更小 LR 是否减轻早期过拟合 | Fold 0 complete；未优于 baseline |
| `lr3e6_bce_b256` | 3e-6 | 256 | BCE | 更快域适配是否提高 minority ranking | Fold 0 complete |
| `lr1e6_bce_b128` | 1e-6 | 128 | BCE | 更频繁、噪声更大的更新是否有利 | Fold 0 complete |
| `lr1e6_bce_b512` | 1e-6 | 512 | BCE | 更平滑更新是否更稳定 | Fold 0 running / GPU1 |
| `lr3e7_wbce_b256` | 3e-7 | 256 | Weighted BCE | 保守 LR 下类别加权 | Fold 0 complete；未优于 baseline |
| `lr1e6_wbce_b256` | 1e-6 | 256 | Weighted BCE | 与 baseline 只改变 loss | Fold 0 complete；未优于 baseline |
| `lr3e6_wbce_b256` | 3e-6 | 256 | Weighted BCE | 较大 LR 与加权梯度的交互 | Pending（排在 b512 后） |

这不是完整 factorial sweep。先做能回答明确问题的一因素筛选，避免同时改变 LR、batch、loss 后无法归因。

2026-08-28 服务器守护任务：GPU0 的 `ecg_lfm_l1a_resume` 顺序运行 baseline folds；GPU1 的 `ecg_lfm_tune_gpu1` 顺序运行 Fold 0 候选；`ecg_lfm_tuning_stages` 等待 Fold 0 队列结束后自动执行晋级与最终冻结流程。守护任务只读内部验证指标进行选择，不读取外部标签。

## 6. 逐级选择规则

1. 所有候选先运行 Fold 0；出现 OOM、NaN、退化概率或明显过拟合者淘汰。
2. 按 record-level macro AUPRC 为第一指标；macro AUROC、NSTEMI/OMI AUPRC、macro F1、Sensitivity、Specificity、PPV 为安全指标。
3. 最佳 unweighted 与最佳 weighted 候选扩展到 folds 1–2，比较三折 mean ± std，避免单折偶然涨点。
4. 三折胜者完成 folds 3–4，得到完整五折 OOF。
5. 只在完整 OOF 上锁定最终逐标签阈值，再对五折 external record-level probabilities 求算术均值。
6. 最终方案只提交外部平台一次；测试指标不用于再次改变方案。

## 7. 曲线与过拟合审计

每个 run/fold 自动保存：

- `training_curves.csv`
- `training_validation_curves.png`
- `overfitting_audit.json`

曲线同时显示 train/validation loss、validation AUROC/AUPRC/F1@0.5、LR、gradient norm。自动审计记录最佳 validation-loss epoch、best 后 validation deterioration、train loss change 和 early-stopping 轨迹。最终 checkpoint 始终由 validation loss 决定；F1@0.5 只作健康监控，最终 F1 使用 record-level OOF 阈值。

Baseline Fold 0–2 已完成；当前 Fold 3 在 epoch 6 达到暂时最佳 validation loss `0.406`（validation AUROC `0.877350`、AUPRC `0.496419`），epoch 8 validation loss 回升至 `0.411`，但仍在 patience 窗口内，不能提前宣布最终 best epoch。train loss 仍下降，过拟合审计将在训练结束后生成。首个 `LR=3e-7` 候选在 epoch 1–2 的 validation AUPRC 为 `0.3467→0.3878`，目前仍处于较慢收敛阶段，尚不能以早期数值淘汰。

### 7.1 2026-08-28 checkpoint 推理恢复事件

Baseline Fold 0 与首个 `lr3e7_bce_b256` 均已正常训练并按 validation loss 保存 strict-best checkpoint，但原推理封装曾清空 checkpoint 中的 `model_path/no_pretrained_weights`，导致官方 loader 以分类器 dataclass 默认的 `1024/12-head` 静态配置重建 encoder，触发 `embed_dim must be divisible by n_heads`。检查 checkpoint state dict 后确认实际训练结构完整：官方 ECG-LFM `24 layers / 16 heads / 4096 FFN`、分类头 `4×1024`、总参数 `622,880,004`。

修复只保留 checkpoint 原始的预训练模型恢复路径，使官方 loader 先按训练时相同的预训练结构重建 encoder，再加载微调 state dict；未修改 checkpoint 权重、数据、模型结构或训练结果，也未重训 Fold 0。CPU 恢复契约已验证 24 层、16 头、四输出分类头和全部参数均可严格加载。补算的 record-level 推理、指标、曲线和 gate 使用该修复逻辑；事件日志保留于 `logs/ecg_lfm/l1a_resume_repaired.log`。

## 8. 最终评价与外部平台

内部每标签及 macro 输出：AUROC、AUPRC、Sensitivity、Specificity、PPV、NPV、F1、MCC、Accuracy、TP/TN/FP/FN；另报告 NSTEMI-vs-UA AUROC/AUPRC/macro-F1/balanced accuracy 和混淆矩阵。

最终外部平台提交后记录：

- submission CSV 及 SHA-256；
- 平台原始 HTML 响应；
- STEMI/NSTEMI/UA/OMI 的 Sensitivity、Specificity、PPV、NPV、Accuracy、F1；
- 与内部 OOF 的泛化差异和可能原因。

平台不返回基于连续概率的 AUROC/AUPRC 时，不伪造外部 AUROC/AUPRC；明确区分本地 OOF ranking metrics 与平台 hard-label metrics。

为落实冻结测试集规则，2026-08-27 已将 baseline 后续流程的 `SUBMIT_EXTERNAL` 默认值设为 `0`，并重启守护任务。Baseline 可以生成外部盲测概率用于检查产物完整性，但不会发起平台 POST；只有内部三折筛选与最终五折 OOF 完成、唯一方案和阈值锁定后，才允许设置提交开关并提交一次。

## 9. 结果表（运行中）

| Run | Macro AUROC | Macro AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy | Overfit audit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline Fold 0 | 0.8214 | 0.4444 | 0.7738 | 0.7558 | 0.3192 | 0.9325 | 0.4444 | 0.3601 | 0.7732 | best epoch 5; overfit flag true |
| Baseline Fold 1 | 0.8245 | 0.4595 | 0.8069 | 0.7294 | 0.3060 | 0.9345 | 0.4263 | 0.3515 | 0.7456 | 已完成；正式 record-level 评价 |
| Baseline Fold 2 | 0.8338 | 0.4740 | 0.8008 | 0.7485 | 0.3183 | 0.9309 | 0.4386 | 0.3633 | 0.7588 | 已完成；正式 record-level 评价 |
| **Baseline Fold 0–2 mean ± SD** | **0.8265 ± 0.0065** | **0.4593 ± 0.0148** | **0.7938 ± 0.0176** | **0.7446 ± 0.0136** | **0.3145 ± 0.0074** | **0.9327 ± 0.0018** | **0.4364 ± 0.0093** | **0.3583 ± 0.0061** | **0.7592 ± 0.0138** | **临时三折结果，不是最终五折** |
| `lr3e6_bce_b256` Fold 0 | 0.8286 | 0.4662 | 0.7729 | 0.7633 | 0.3311 | 0.9348 | 0.4567 | 0.3726 | 0.7816 | best epoch 3; overfit flag true |
| `lr1e6_bce_b128` Fold 0 | 0.8196 | 0.4381 | 0.8231 | 0.7078 | 0.2931 | 0.9371 | 0.4205 | 0.3431 | 0.7350 | best epoch 3; overfit flag true |
| `lr3e7_bce_b256` Fold 0 | 0.8198 | 0.4342 | 0.8179 | 0.7061 | 0.2953 | 0.9404 | 0.4255 | 0.3442 | 0.7409 | best epoch 14; overfit flag true |
| `lr3e7_wbce_b256` Fold 0 | 0.8149 | 0.4123 | 0.8258 | 0.7000 | 0.3049 | 0.9394 | 0.4337 | 0.3502 | 0.7370 | best epoch 13; overfit flag true |
| `lr1e6_wbce_b256` Fold 0 | 0.8161 | 0.4156 | 0.8325 | 0.6950 | 0.2935 | 0.9414 | 0.4245 | 0.3441 | 0.7334 | best epoch 4; overfit flag true |
| Batch candidate | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Final selected | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

候选方案与 baseline 的首轮消融行均为同一 Fold 0、record-level、各自在该验证折上搜索的 Youden 阈值；Baseline Fold 1–2 和三折均值是随后完成的阶段性结果。`lr3e7_bce_b256` 在 Fold 0 相对 baseline：macro-AUPRC `-0.0102`、Sensitivity `+0.0441`，但 Specificity `-0.0497`、F1 `-0.0189`。两个 Weighted BCE 的 macro-AUPRC 分别下降 `0.0321` 与 `0.0288`；Sensitivity 分别提高 `0.0521` 与 `0.0587`，但 Specificity 分别下降 `0.0558` 与 `0.0607`，F1 也均下降。当前没有证据表明训练折比例加权能够提高总体少数类排序性能，仍需等待其余 LR/batch 候选完成后再锁定方案。

### 9.1 ECG-LFM baseline 与历史 E4 的阶段性比较（2026-08-28）

两者使用相同的 17,960 条内部 ECG、17,018 名患者、四个独立 sigmoid 标签、历史 E4 patient-level 五折索引和 seed 2026。ECG-LFM 为官方 24-layer/16-head、622,880,004 参数模型；E4 为 Xception morphology + LeadSpatial GAT + WavelNet frequency expert + task routers + 4 个 KAN heads，共 382,446 参数。ECG-LFM 参数量约为 E4 的 1,628.7 倍。两套方案的输入分辨率、预处理、训练协议和 checkpoint 选择规则并不完全相同，因此这是完整方案比较，不是只改变网络结构的严格消融。

不依赖分类阈值的同折 ranking 指标如下。当前只比较已完成的相同 Fold 0–2：

| Fold | ECG-LFM AUROC | E4 AUROC | Δ | ECG-LFM AUPRC | E4 AUPRC | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8214 | 0.8131 | +0.0082 | 0.4444 | 0.4428 | +0.0016 |
| 1 | 0.8245 | 0.8190 | +0.0055 | 0.4595 | 0.4465 | +0.0131 |
| 2 | 0.8338 | 0.8238 | +0.0100 | 0.4740 | 0.4568 | +0.0172 |
| **Fold 0–2 mean ± SD** | **0.8265 ± 0.0065** | **0.8186 ± 0.0054** | **+0.0079** | **0.4593 ± 0.0148** | **0.4487 ± 0.0072** | **+0.0106** |

按四个标签拆分的阶段性结果（Fold 0–2 mean ± SD）如下；这里的 AUROC/AUPRC 是阈值无关的直接比较：

| 标签 | ECG-LFM AUROC | E4 AUROC | ECG-LFM AUPRC | E4 AUPRC | ECG-LFM Sensitivity | E4 Sensitivity | ECG-LFM Specificity | E4 Specificity | ECG-LFM F1 | E4 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STEMI | 0.9484 ± 0.0070 | 0.9357 ± 0.0135 | 0.6889 ± 0.0324 | 0.6744 ± 0.0307 | 0.8856 ± 0.0178 | 0.8081 ± 0.0599 | 0.8867 ± 0.0073 | 0.9013 ± 0.0227 | 0.5563 ± 0.0175 | 0.5512 ± 0.0199 |
| NSTEMI | 0.7867 ± 0.0144 | 0.7864 ± 0.0108 | 0.2585 ± 0.0113 | 0.2548 ± 0.0026 | 0.6974 ± 0.0584 | 0.7203 ± 0.0567 | 0.7588 ± 0.0437 | 0.7274 ± 0.0489 | 0.2811 ± 0.0149 | 0.2663 ± 0.0126 |
| UA | 0.6705 ± 0.0048 | 0.6624 ± 0.0100 | 0.4834 ± 0.0097 | 0.4767 ± 0.0120 | 0.7080 ± 0.0349 | 0.7531 ± 0.1191 | 0.5441 ± 0.0331 | 0.4730 ± 0.1293 | 0.5504 ± 0.0093 | 0.5461 ± 0.0183 |
| OMI | 0.9006 ± 0.0083 | 0.8902 ± 0.0076 | 0.4065 ± 0.0367 | 0.3889 ± 0.0150 | 0.8842 ± 0.0544 | 0.8047 ± 0.0877 | 0.7888 ± 0.0402 | 0.8232 ± 0.0491 | 0.3579 ± 0.0232 | 0.3703 ± 0.0273 |

当前工作点指标的三折描述性均值如下：

| 方案（Fold 0–2） | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| ECG-LFM baseline | 0.7938 | 0.7446 | 0.3145 | 0.9327 | 0.4364 | 0.3583 | 0.7592 |
| E4 | 0.7715 | 0.7312 | 0.3150 | 0.9330 | 0.4334 | 0.3457 | 0.7531 |
| **ECG-LFM − E4** | **+0.0223** | **+0.0134** | **−0.0005** | **−0.0003** | **+0.0030** | **+0.0126** | **+0.0061** |

阈值相关表只能作阶段性描述：ECG-LFM 当前每折在本折搜索 Youden 阈值，而 E4 折指标使用完整 pooled OOF 上锁定的一组全局 Youden 阈值，二者尚不是完全等价的阈值评价。E4 完整五折的历史结果为 macro AUROC `0.8208 ± 0.0049`、AUPRC `0.4503 ± 0.0076`、Sensitivity `0.8045 ± 0.0587`、Specificity `0.6983 ± 0.0568`、F1 `0.4147 ± 0.0256`。当前 ECG-LFM 前三折在 AUROC/AUPRC 上每折均高于 E4 同折，但增幅较小，尚不能据此宣布最终优于 E4。

最终判断必须等待 ECG-LFM Fold 3–4、完整 17,960 条 pooled OOF 和一次全局阈值锁定完成，再运行 `scripts/compare_ecg_lfm_to_e4.py`。该脚本会核对相同 record ID、patient ID 和标签，并进行 patient-cluster paired bootstrap；外部泛化目前也无法比较，因为 E4 已有平台结果，而 ECG-LFM 尚未冻结方案或提交平台。

### 9.2 当前已完成验证结果中的最高单折（四标签）

在当前四标签多标签二分类的已完成结果中，按预注册主指标 macro-AUPRC 排名最高的是 `finetune_paper / Fold 2`：macro AUROC `0.8338`、macro AUPRC `0.4740`、macro F1 `0.4386`。该结果来自单个内部验证折（3,596 条 record），不是五折 pooled OOF，不能作为最终泛化指标。记录文件为远端 `results/ecg_lfm_4label/finetune_paper/fold2/evaluation/fold_evaluation.json`，本折 Youden 阈值为 `STEMI=0.05`、`NSTEMI=0.07`、`UA=0.37`、`OMI=0.03`。

| 标签 | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STEMI | 0.9552 | 0.7257 | 0.8958 | 0.8945 | 0.4250 | 0.9900 | 0.5765 | 0.5727 | 0.8946 |
| NSTEMI | 0.8033 | 0.2516 | 0.7459 | 0.7342 | 0.1696 | 0.9754 | 0.2764 | 0.2639 | 0.7350 |
| UA | 0.6696 | 0.4725 | 0.6707 | 0.5719 | 0.4516 | 0.7677 | 0.5398 | 0.2307 | 0.6060 |
| OMI | 0.9069 | 0.4462 | 0.8908 | 0.7936 | 0.2269 | 0.9907 | 0.3617 | 0.3860 | 0.7998 |
| **Macro** | **0.8338** | **0.4740** | **0.8008** | **0.7485** | **0.3183** | **0.9309** | **0.4386** | **0.3633** | **0.7588** |

对应混淆矩阵（TP/TN/FP/FN）为：STEMI `258/2959/349/30`，NSTEMI `182/2461/891/62`，UA `831/1348/1009/408`，OMI `204/2672/695/25`。调参候选中 `lr3e6_bce_b256` 的单折 macro-F1 为 `0.4567`，但 macro-AUPRC `0.4662` 低于该 baseline Fold 2，且过拟合审计为 true；因此当前不能按 F1 单项把它定为最终方案。

## 10. 代码和产物

- `experiments/ecg_lfm/run_fold.py`：训练折专属权重和 provenance。
- `experiments/ecg_lfm/contract_tests.py`：unweighted/weighted BCE 数值等价及五折 leakage contract。
- `scripts/run_ecg_lfm_tuning_fold0.sh`：Fold 0 候选队列。
- `scripts/run_ecg_lfm_tuning_multifold.sh`：优胜候选的 Fold 1–2 扩展、最终 Fold 3–4 和五折产物生成；该脚本不自动提交外部平台。
- `scripts/evaluate_ecg_lfm_fold.py`：单折 record-level 全指标。
- `scripts/plot_ecg_lfm_training_curves.py`：曲线和过拟合审计。
- `scripts/summarize_ecg_lfm_multifold.py`：已完成折的 `mean ± std` 比较表。
- `results/ecg_lfm_4label_tuning/`：所有候选独立结果目录。
- `logs/ecg_lfm_tuning/`：队列日志。

旧 E4 与原 ECG-LFM baseline 目录均只读保留，不会被调参分支覆盖。

可选 L0 linear probe 已完成健康验证，但排序指标远低于 L1a。为落实新的调参优先级，L0 在 epoch 18 后停止并释放 GPU1；其最后中间 AUROC/AUPRC 为 `0.5862/0.1565`，checkpoint/log 均保留，但明确标记为 incomplete diagnostic，不进入完成结果表。取证文件：`artifacts/ecg_lfm/linear_probe_fold0_reprioritization.json`。

<!-- AUTO_RESULTS_START -->

## 11. 自动更新的最终实验结果

当前实验仍在运行；本区块将在三折选择、最终五折和外部平台提交完成后由可追溯脚本更新。

<!-- AUTO_RESULTS_END -->
