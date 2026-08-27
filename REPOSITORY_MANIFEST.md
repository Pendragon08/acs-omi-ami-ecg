# 仓库整理清单

## 纳入 GitHub 仓库

| 类别 | 来源 | 目标位置 |
|---|---|---|
| 最终与历史实验代码 | .remote_project_scan 的 source directories | experiments、scripts、preprocessing、data_provider、losses、acs_utils、nets、exp |
| 基线代码 | ACS-Baseline-main 中的 Python 源码与 README | baselines/ACS-Baseline-main |
| 配置 | 根目录 configs 与 .remote_project_scan/configs | configs |
| 方案报告 | 根目录三份 ACS 方案和改进提示词 | docs/plans |
| 实验报告 | reports、FINAL_REPORT、loss/feature ablation 报告 | docs/reports |
| 聚合实验产物 | 不含逐例信息的 summary、history、label audit、threshold、平台指标 | results |

## 明确排除

| 内容 | 原因 |
|---|---|
| med_data、任何 .med、.npy、.npz、data/ | 受控临床数据或波形，待数据治理授权 |
| train.csv、test.csv、逐例 OOF/测试/提交 CSV、split.json | 包含或可关联患者/ECG 标识 |
| checkpoint .pt | 体积大且不属于当前安全发布的最小复现包 |
| ACS-Baseline-main.zip、med_data.zip | 与目录内容重复且超过普通 GitHub Git 文件大小限制 |
| .ssh_gpu_check_tmp、.tmp_e4_npz2、.remote_exec_tmp.js、.agents | 临时工具、依赖缓存或执行环境文件 |
| agent-frontend_PR15_README.md、lithium_battery_SOH_RUL_2024_2026_Q1_FULL_research_gaps_updated_20260814.md | 与 ACS ECG 研究无关 |

## 版本与审计

当前四标签实验使用的代码、配置、训练/测试 CSV、OOF、阈值、提交文件和平台回包的 SHA-256 记录在 docs/reports/e4_stemi_nstemi_ua_omi_multilabel_provenance_20260827.json。

仓库默认保持 private。是否纳入受控数据、逐例结果和权重，须先满足 DATASET.md 的发布前提。

