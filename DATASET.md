# 数据集说明与发布边界

## 标签定义

源标签包含 `STEMI`、`NSTEMI`、`UA`、`OMI`。前三者是互斥的 ACS 亚型，不能
使用三个彼此独立的 sigmoid 作为最终 ACS 分类器。当前临床评价包含：

- ACS 阳性样本的 STEMI/NSTEMI/UA 单标签三分类；
- 与 ACS 亚型独立的 OMI/NOMI 二分类。

源数据中 `[0,0,0]` 是明确的 non-ACS，而不是缺失标签。全队列一阶段模型因此
保留 `OTHER` 状态，但报告不得把 OTHER 称为第四种 ACS 亚型。任何多个 ACS
亚型同时为 1、标签非二值或记录标识重复冲突都应作为数据错误终止运行。

## 数据规模

| 项目 | 数量 |
|---|---:|
| 训练 ECG | 17,960 |
| 训练患者 | 17,018 |
| non-ACS/OTHER | 9,070 |
| STEMI | 1,442 |
| NSTEMI | 1,235 |
| UA | 6,213 |
| OMI 阳性 | 1,151 |
| 外部盲测 ECG | 1,995 |

患者级五折使用 seed=2026。每位患者只能出现在一个验证折，五个验证折应精确
覆盖全部训练记录。训练增强和 scaler 拟合仅发生在训练折；外部集不参与
checkpoint、超参数或阈值选择。

## 输入管线

- E4 和传统网络使用既有 P2 预处理数组及训练折内 robust scaling；
- ECG-LFM 使用官方 500 Hz、12 导联输入规范和预训练权重；
- 每条 10 秒 ECG 划分为两个 5 秒 segment，记录级预测取两个 segment logit
  的平均；
- E4 的 Frequency Expert 使用端到端 WavelNet，不使用 MVMD 作为网络外部预处理。

## 不进入 GitHub 的内容

- 原始或预处理 ECG 波形；
- train/test CSV、患者级表格、fold 索引和逐例标识符；
- OOF、验证或外部逐例概率/预测与提交 CSV；
- checkpoint、预训练权重和原始 Fairseq/screen 日志；
- 密码、访问令牌、SSH 私钥及其他凭据。

GitHub 只保存代码、配置、聚合指标、脱敏训练曲线、图表和报告。完整规则见
[REPOSITORY_MANIFEST.md](REPOSITORY_MANIFEST.md) 与 `.gitignore`。
