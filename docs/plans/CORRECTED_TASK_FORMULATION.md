# 当前唯一任务与实施规范

生效日期：2026-08-31。

## 1. 任务定义

本项目是联合多任务分类，不是四个独立标签的纯多标签分类：

1. ACS 阳性亚型：`STEMI / NSTEMI / UA` 互斥单标签三分类；
2. 病理实质维度：`OMI / NOMI` 独立二分类。

全队列中的 `[0,0,0]` 是 non-ACS。当前一阶段实现为了保留这些记录，增加
`OTHER` 工程状态并训练四状态 softmax；主报告仍将 ACS 临床端点定义为限定
ACS 阳性后的三分类。

## 2. 模型输出与损失

- 状态头：`OTHER/STEMI/NSTEMI/UA` 四个 logit，交叉熵训练，softmax+argmax；
- OMI 头：一个 logit，数值稳定的 `BCEWithLogitsLoss`，sigmoid 推理；
- 总损失：`L = L_state_CE + lambda * L_OMI_BCE`；
- 权重只能由当前训练折统计；验证/外部集禁止参与权重计算；
- ACS 不使用 Youden 阈值；只有 OMI 可以在验证集选择阈值。

若后续改为两阶段模型，应先独立判断 ACS/non-ACS，再只对 ACS 阳性进行三分类，
并重新训练和评价；不得把当前一阶段结果直接当作两阶段结果。

## 3. 数据与划分

- 17,960 条训练 ECG、17,018 位患者；
- 患者级五折，seed=2026；
- 每折训练/验证患者交集必须为 0，五个验证折精确覆盖训练池；
- `[0,0,0]` 保留，亚型多阳性、重复冲突或非二值标签立即报错；
- scaler、augmentation、类别权重只使用训练折；
- 外部 1,995 条 ECG 只在方案冻结后推理一次。

## 4. 评价

ACS 阳性三分类主指标：Macro-F1、Accuracy、三分类混淆矩阵；有连续概率时再
报告 macro one-vs-rest AUROC/AUPRC。OMI 报告 AUROC、AUPRC、Sensitivity、
Specificity、PPV、NPV、F1、MCC、Accuracy 和 TP/TN/FP/FN。

OMI 同时记录固定 0.5、最大 F1、Youden J 和敏感度约束工作点。所有阈值必须
在内部验证数据确定后冻结。平台只返回逐类别 hard-label 指标时，不得将其称为
联合三分类总指标。

## 5. 当前实现入口

- E4：`experiments/acs_joint_e4.py`；
- 传统模型：`experiments/acs_joint_model_comparison.py`；
- ECG-LFM：`experiments/ecg_lfm/run_joint_fold.py` 与 `evaluate_joint.py`；
- 配置：`configs/ecg_lfm/corrected_joint_*.yaml`；
- 外部提交：`scripts/submit_joint_acs_omi.py`。

## 6. 解释限制

当前 pooled OOF 使用验证损失选择 checkpoint，并在同一 pooled OOF 上选择 OMI
阈值后回算硬指标，因此工作点指标属于调优后的描述性结果。论文确认性实验应
采用嵌套患者级校准或预先固定阈值；未完成配对统计检验时不得宣称统计显著。
