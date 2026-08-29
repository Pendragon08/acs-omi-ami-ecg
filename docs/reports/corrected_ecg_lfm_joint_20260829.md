# ECG-LFM 校正联合任务实验报告

更新时间：2026-08-29（UTC）
状态：GPU1 上运行中，尚未形成可引用的最终指标。

## 任务定义

ECG-LFM 当前使用五个输出：ACS 四分类
`OTHER / STEMI / NSTEMI / UA`，以及独立 OMI/NOMI 二分类。STEMI、NSTEMI、
UA 在 ACS 头中使用交叉熵和 softmax-argmax，不能同时为阳性；OMI 使用稳定的
`BCEWithLogitsLoss` 和验证集确定的阈值。`[0,0,0]` 保留为 `OTHER`，而不是
缺失值。该定义与 E4、传统架构完全一致，保证比较只反映编码器/微调方式差异。

## 代码与错误修复记录

首次校正运行在 Fairseq 验证阶段暴露真实的形状错误：源标签是四列
`[STEMI,NSTEMI,UA,OMI]`，而模型 `num_labels=5`，验证循环 reshape 失败。
该运行没有有效指标，不能作为结果。现在的
`experiments/ecg_lfm/run_joint_fold.py` 会在受控执行环境生成并审计私有的
五列 `[OTHER,STEMI,NSTEMI,UA,OMI]` 标签文件；criterion
`third_party/ECG-LFM/fairseq_signals/criterions/acs_joint.py` 同时兼容源四列
检查格式和 Fairseq 五列格式，并对非二值标签、亚型重叠、缺失 OTHER 状态直接
报错。修复后 fold0 的验证 reshape smoke test 已通过。

## 固定实验配置

| 项目 | 配置 |
|---|---|
| 划分 | 历史患者级五折，seed=2026，不重新随机划分 |
| 适配方式 | Baseline、LoRA、DoRA，各 5 折 |
| GPU | 物理 GPU1（`CUDA_VISIBLE_DEVICES=1`） |
| batch | 64；`update_freq=4`，有效 batch size=256 |
| 优化器 | Adam，沿用各适配方式登记的学习率 |
| 损失 | ACS CE + OMI 稳定 BCE，未叠加额外 `pos_weight` |
| 训练上限 | 100 epoch，Fairseq patience=10，逐 epoch 验证 |
| 评估 | ACS 四分类、ACS 阳性条件三分类、OMI 二分类；OMI 阈值只在验证 OOF 上确定 |

运行根目录为 `results/corrected_ecg_lfm_joint_bs64_v2/`，原始日志目录为
服务器上的 `logs/corrected_ecg_lfm_joint_bs64_v2/`。训练脚本使用 screen，并且
只绑定一个 GPU。

## 当前进度（报告更新时间）

Baseline fold0 已完成到第 6 个 epoch；最近一次记录为验证总损失 1.07442、
ACS accuracy 0.56004、OMI accuracy 0.93275，训练学习率约 `9.41e-7`。其余
fold 和 LoRA/DoRA 尚未完成，因此这里不报告“最佳 ECG-LFM 指标”，也不把部分
epoch 当作模型比较结果。完成后必须等 15 个运行和统一评估器全部结束，才会
更新本报告的 OOF、五折均值±标准差、混淆矩阵和外部平台逐标签指标。

## 结果归档计划

每个适配方式完成后将写入：

- `summary/summary.json`、五折 OOF/阈值/外部平台聚合指标；
- 每折训练历史和脱敏的 `training_log_sanitized.csv`（仅 fold、epoch、loss、
  验证指标、学习率）；
- 训练/验证损失与学习率、ROC/PR、混淆矩阵 PNG；
- 外部平台返回的聚合 JSON 与提交 provenance（不含提交 CSV）。

原始 ECG、患者级表格、checkpoint、逐例预测、Fairseq 原始 JSONL 和服务器
路径不会上传 GitHub。可公开的脱敏日志和图表会在队列完成后追加到
`docs/reports/实验结果总报告_20260829.md`，并在
`docs/logs/EXPERIMENT_LOG_INDEX.md` 登记。
