# 用于让 ChatGPT 改进 ACS/OMI 最终方案的提示词（任务定义已校正）

> **重要任务定义（2026-08-30）**：本文早期的 OMI/AMI 双独立分类器描述是
> 历史背景，不是当前主任务。当前应按 `CORRECTED_TASK_FORMULATION.md`
> 执行：ACS `OTHER/STEMI/NSTEMI/UA` 互斥单标签四分类 + 独立 OMI/NOMI
> 二分类。不要把 STEMI、NSTEMI、UA、OMI 当成四个独立 sigmoid 标签。

请你作为一名熟悉医学 ECG 深度学习、患者级数据划分、类别不平衡和临床模型评价的研究者，基于下面已经完成的 ACS ECG OMI/AMI 二分类方案，设计一份**可执行、可复现、严格避免数据泄漏的改进方案**。

不要凭空补充数据集来源、患者人口学信息、标签定义或外部测试集标签；没有提供的信息请明确写成“未知/需要核实”。不要只给泛泛的建议，要给出具体的数据流、模型、损失函数、训练流程、阈值策略、实验矩阵、统计分析和最终报告格式。

## 一、当前项目和任务定义（校正版）

项目使用现有 ACS 预处理 ECG 数据，目标是从 12 导联 ECG 同时完成两个相关
输出任务：

1. ACS 亚型：`OTHER / STEMI / NSTEMI / UA` 四分类，其中后三者互斥；
2. OMI（occlusion myocardial infarction，闭塞性心肌梗死）/NOMI 二分类，
   可与任意 ACS 状态共存。

当前实际实施的是一个**混合输出联合/多任务模型**：ACS 头输出 4 个
softmax logits，OMI 头输出 1 个 sigmoid logit（总计 5 个 logits）。每个模型
输入一条 10 秒、100 Hz、12 导联 ECG，原始张量形状为 `[1000, 12]`。源
`[0,0,0]` 映射为 `OTHER`，不能删除；`OTHER + OMI` 是有效组合。

## 二、数据集情况

- 官方训练集：17,960 条 ECG 记录，包含 ACS 亚型和 OMI 标签；OMI 阳性数为
  1,151（约 6.4%）。ACS 状态为 OTHER 9,070、STEMI 1,442、NSTEMI 1,235、
  UA 6,213。
- 官方外部测试集：1,995 条 ECG 记录，标签隐藏，只能在方案冻结后进行一次最终推理/平台评估。
- 历史固定划分：按照 `patient_id` 做患者级五折划分，seed=2026；同一患者的记录不能跨集合，OOF 覆盖全部 17,960 条训练池记录。
- 每折验证约 3,600 条记录；不使用外部测试集参与训练、选 checkpoint、调阈值或选择方案。
- 源 `[0,0,0]` ACS 状态是明确的 OTHER；其中包含 1 条 `ACS=0, OMI=1` 的有效组合。
- 外部测试集的真实标签不可见，因此不能使用它调参、选阈值、选择模型或反复试验后挑最好结果。

请在修改方案时保留这份历史划分作为可比基线，并明确说明：训练集、验证集和外部测试集分别承担什么作用；是否需要重新生成严格的患者级分层划分；如何保存 split 文件及其 SHA-256；如何核查患者和记录没有交叉。

## 三、历史 OMI/AMI 基线方案（仅作溯源；不代表当前主任务）

> 本节原有的两个独立 OMI/AMI 二分类器及其数值保留作历史背景。当前主任务
> 的模型、损失和指标必须按上方校正版任务定义及 `CORRECTED_TASK_FORMULATION.md`
> 执行。

### 1. 预处理 P3

当前数据流为：

```text
原始 ECG [1000, 12]
 -> 0.5–40 Hz、4 阶、零相位 Butterworth 带通滤波（fs=100 Hz，sosfiltfilt）
 -> 全局稳健缩放：median + 1.4826 × MAD
 -> 训练集在线增强：高斯噪声 p=0.3、幅度缩放 p=0.3、时间平移 p=0.2
 -> XceptionTime
```

历史 P3 的 scaler 曾在官方训练全集（17,960 条，包含内部 validation）上拟合。这是一个需要修正的方法学问题：改进方案必须只使用 train 子集拟合 median/MAD，然后固定该 scaler 变换 validation 和外部测试；禁止使用 validation/test 的任何统计量拟合预处理器。

外部测试推理不使用随机增强。验证集也不使用 oversampling、随机增强或其他改变分布的操作。

### 2. 模型

当前最终模型为 **XceptionTime**，不是 TimesNet：

- InceptionTime 风格的 6 个多尺度卷积块；
- 每个块使用 bottleneck `1×1 Conv` 降到 32 通道；
- 三个 depthwise-separable 分支，卷积核为 39、19、9；
- 一个最大池化分支；四个分支拼接为 128 通道；
- 两个残差快捷连接；
- 全局平均池化得到 128 维表示；
- 线性分类头输出一个 logit；
- 约 93,121 个可训练参数；
- 输入由 `[B,1000,12]` 转为卷积需要的 `[B,12,1000]`。

OMI 和 AMI 分别训练一个 XceptionTime 二分类模型，共享架构但不强制共享权重。

### 3. 损失、优化和训练

- loss：无类别权重的 `BCEWithLogitsLoss`；
- 模型输出 raw logits，推理时才使用 sigmoid；
- optimizer：Adam；
- OMI：learning rate = 3e-4，batch size = 256；
- AMI：learning rate = 1e-4，batch size = 128；
- 最大 epoch = 100；
- early stopping patience = 15；
- gradient clipping：max norm = 1.0；
- random seed = 2026；
- 当前实现没有实际使用学习率 scheduler，不能在报告中误写成使用了 scheduler。

### 4. 当前阈值和评价

当前最终报告使用 validation 上的 `argmax F1` 选择阈值，搜索范围约为 0.05–0.95：

- OMI threshold = 0.48；
- AMI threshold = 0.42。

当前 validation 指标：

| 任务 | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | Accuracy | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OMI | 0.8813 | 0.3493 | 0.5319 | 0.9634 | 0.4464 | 0.9737 | 0.4854 | 0.9406 | 0.48 |
| AMI | 0.8760 | 0.6283 | 0.6132 | 0.9462 | 0.6422 | 0.9395 | 0.6274 | 0.9009 | 0.42 |

官方外部测试集标签不可见，现有平台反馈为：

| 任务 | Sensitivity | Specificity | PPV | NPV | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| OMI | 0.5000 | 0.9614 | 0.4706 | 0.9656 | 0.4848 | 0.9318 |
| AMI | 0.6455 | 0.9393 | 0.6520 | 0.9376 | 0.6487 | 0.8952 |

这些平台分数应标注为 development-leaderboard/官方测试反馈，而不是声称为完全盲法、独立且首次获得的外部验证结果。

## 四、已有实验结论

已经做过的对照表明：

- XceptionTime + P3 + BCE 是当前验证集综合表现最好的最终基线；
- Focal、类别平衡 Focal、Weighted BCE 没有在 5 个 seed 上稳定提高 OMI 的 AUPRC、F1 和 PPV；CB-Focal 不能直接作为论文核心创新；
- 将 median ECG、年龄、性别加入融合模型，未带来稳定的 OMI/AMI AUPRC 或 F1 提升；
- TimesNet 等长时序 Transformer 在当前数据和训练设置下不如轻量 1D CNN；
- 不能仅因为 Accuracy 增加就认定改进成功；OMI 少数类漏诊（FN）必须重点关注。

## 五、请你设计的改进目标

改进目标不是盲目追求单一 Accuracy，而是优先提高 OMI 的：

1. Sensitivity/Recall；
2. AUPRC；
3. F1；
4. PPV；
5. AUROC；

同时尽量保持 Specificity，不接受 Sensitivity 提升但 Specificity 大幅下降的“伪提升”。AMI 也要报告同样的完整指标，但 OMI 漏诊是首要临床风险。

## 六、改进方案必须遵守的原则

1. 首先给出“当前基线 vs 改进方案”的差异表，不要直接改模型而不说明改了什么。
2. 修复 P3 scaler 的数据泄漏：median/MAD 只能由 train 子集拟合。
3. 保留 patient-level split；如果提出新的 split，必须和历史 split 并行报告，不能覆盖历史结果。
4. 训练、验证、外部测试的预处理必须严格一致，只有训练集可以做随机增强。
5. 所有 loss、采样、增强、模型结构的改变必须拆成独立消融实验，不能同时改变多个因素后宣称某个因素有效。
6. 不能把 CB-Focal 和 WeightedRandomSampler、正类 oversampling、pos_weight 重复叠加，除非单独设计并明确标注实验。
7. threshold 只能在 validation 或嵌套 validation 中确定，不能在外部测试集上寻找最佳阈值。
8. 对 OMI 少数类同时报告 AUROC、AUPRC、Sensitivity、Specificity、PPV、NPV、F1、MCC、Accuracy、TP、TN、FP、FN。
9. 至少使用 seed=42、2024、2025、2026、3407，报告 mean ± std；不能只挑最好 seed。
10. 最终外部测试只能在方案、阈值和 checkpoint 冻结后评估一次。
11. 不要编造不存在的外部测试标签、置信区间、统计显著性或数据来源。

## 七、建议的改进实验矩阵

请优先设计以下阶段，并说明每阶段的选择标准：

### 阶段 A：严格重建无泄漏基线

- XceptionTime + P3(train-only scaler) + BCE；
- 历史 split 与严格无泄漏 split 都记录；
- 5 个 seed；
- 保持现有模型、Adam、学习率、batch size、epoch、early stopping 和增强配置不变；
- 核对重建后的 AUROC/AUPRC/F1/Sensitivity/Specificity 与历史结果差异。

### 阶段 B：阈值策略对比

在 validation 上预先定义并固定以下阈值：

- threshold=0.50；
- Youden index 最大值；
- F1 最大值；
- F1 与 Sensitivity 的调和均值最大值；
- Sensitivity ≥ 0.90 时 Specificity 最大；
- Sensitivity ≥ 0.95 时 Specificity 最大（若可行）。

必须给出每个阈值策略在 validation 和最终外部测试上的结果，并明确哪个阈值是论文主分析、哪个是临床高敏感性补充分析。

### 阶段 C：单因素 loss 消融

在完全相同数据、split、seed、模型、优化器、学习率和训练流程下比较：

- BCE baseline；
- train-only `pos_weight` Weighted BCE；
- standard Focal Loss；
- Class-Balanced Focal Loss；
- 如需增加 ASL，必须作为额外独立组。

CB-Focal 的 beta/gamma 只能用 train 类别数和 validation 选择，不能使用外部测试标签。若没有跨 seed 的稳定提升，应明确写“无说服力提升”，不能继续挑参直到出现正结果。

### 阶段 D：特征消融

只在模型和训练设置固定时比较：

- raw ECG；
- raw ECG + age；
- raw ECG + sex；
- raw ECG + age + sex；
- raw ECG + median ECG；
- raw ECG + median ECG + age + sex。

必须报告单 seed 筛选和 5-seed 确认两层结果，并说明年龄、性别和 median ECG 是否真正带来稳定增益。

### 阶段 E：模型结构消融（只有在必要时）

若建议更换模型，至少比较 XceptionTime、ResNet1D、InceptionTime 和 TimesNet，并控制参数量、训练预算、split、seed 和阈值策略。请说明为什么改模型，不能把模型、loss、特征和采样同时改动。

## 八、统计和报告要求

请输出一份完整的修改后方案，至少包含：

1. 研究问题与主要假设；
2. 数据集、标签、记录数、患者数、阳性率和外部测试集说明；
3. 患者级划分和防止泄漏的核查方法；
4. 预处理 P0/P3 的数学定义和 train-only scaler 实现；
5. XceptionTime 结构、输入输出维度和参数量；
6. loss、optimizer、学习率、batch size、scheduler、epoch、early stopping、增强和随机种子；
7. 阈值选择流程；
8. 完整实验矩阵和公平比较原则；
9. 评价指标和混淆矩阵定义；
10. 5-seed mean ± std；
11. Baseline 与最佳方案的 bootstrap 95% CI 和 paired bootstrap（如果数据允许）；
12. OMI 和 AMI 的最终结果表；
13. 失败实验和负结果；
14. 局限性：类别不平衡、患者级划分、scaler 历史泄漏修复、外部测试标签不可见、平台反馈不是完全盲法验证；
15. 复现命令、配置文件、checkpoint、预测文件和结果目录结构。

最终请明确回答：

- 修改后的最终模型到底是什么？
- 是否仍然使用 XceptionTime，还是有充分证据更换为其他模型？
- 是否仍然使用纯 raw ECG，还是加入年龄/性别/median ECG？
- 最终 loss 是什么？
- OMI 和 AMI 的最终阈值是多少？
- 各指标 mean ± std 是多少？
- 是否在多个 seed 上稳定？
- 是否有统计学证据支持改进？
- 是否值得作为 SCI 论文方法学创新点？

请把结论分为 `Strong improvement`、`Moderate improvement` 或 `No convincing improvement`，并且允许结论为负面结果。任何不能由现有数据验证的内容都要标注为“待实验验证”，不要把计划写成已经完成的结果。
