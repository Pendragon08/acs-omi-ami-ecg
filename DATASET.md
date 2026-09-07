# 数据集说明与发布边界

_更新于 2026-09-07 · 当前主任务为 Joint-6；旧双 Head / E4 仅为历史方案_

---

## 🎯 标签定义

源标签包含 `STEMI`、`NSTEMI`、`UA`、`OMI`。前三者是互斥的 ACS 亚型，不能
使用三个彼此独立的 sigmoid 作为最终 ACS 分类器。当前模型使用单一 Joint-6
Softmax + CE：`OTHER / STEMI+nOMI / STEMI+OMI / NSTEMI+nOMI / NSTEMI+OMI / UA`。
从六类概率恢复 ACS 四状态概率和 OMI 二分类概率；这不表示模型仍使用旧双 Head。

源数据中 `[0,0,0]` 是明确的 non-ACS，而不是缺失标签。全队列一阶段模型因此
保留 `OTHER` 状态，但报告不得把 OTHER 称为第四种 ACS 亚型。任何多个 ACS
亚型同时为 1、标签非二值或记录标识重复冲突都应作为数据错误终止运行。

唯一 `OTHER+OMI` 训练时映射为 OTHER，但原始 OMI 阳性标签保留用于 OMI 主评价；UA+OMI 为 0，因此不建立该空类。[Joint-6 标签审计](results/joint6_cv/label_audit/)保存安全聚合计数。OTHER 不是第四种 ACS 亚型；仅在真实 ACS 子集中，亚型才是 STEMI/NSTEMI/UA 三分类。

## 📊 数据规模

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

患者级 outer 五折使用 seed=2026。每位患者只能出现在一个 outer 验证折，五个折
精确覆盖全部训练记录；每个 outer-train 内再划患者互斥的 inner-train / inner-validation。
checkpoint 与 OMI 阈值由 inner-validation 决定，outer-test 只做冻结后评价。外部集不参与
checkpoint、超参数或阈值选择，当前 Joint-6 尚无外部测试指标。

## ⚙️ 输入管线

- 当前传统网络使用既有 P2 预处理数组及训练折内 robust scaling；
- ECG-LFM 使用官方 500 Hz、12 导联输入规范和预训练权重；
- 每条 10 秒 ECG 划分为两个 5 秒 segment，记录级预测取两个 segment logit
  的平均；
- 历史 E4 使用 P2 与端到端 WavelNet，不使用 MVMD 外部预处理；E4 不是当前 Joint-6 主模型。

两条输入管线不同，所以传统模型与 ECG-LFM 对比是工程方案比较，不是仅改变模型结构的单变量消融。

## 🔐 不进入 GitHub 的内容

- 原始或预处理 ECG 波形；
- train/test CSV、患者级表格、fold 索引和逐例标识符；
- OOF、验证或外部逐例概率/预测与提交 CSV；
- checkpoint、预训练权重和未经审阅的原始 Fairseq/screen 日志；
- 密码、访问令牌、SSH 私钥及其他凭据。

GitHub 保存代码、配置、聚合指标、脱敏训练曲线、图表和报告；既有总调度日志与部分历史聚合 RL 日志的保留范围见[日志索引](docs/logs/EXPERIMENT_LOG_INDEX.md)。完整规则见
[REPOSITORY_MANIFEST.md](REPOSITORY_MANIFEST.md) 与 `.gitignore`。
