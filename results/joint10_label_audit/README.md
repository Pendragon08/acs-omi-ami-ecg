# Joint-10 标签可行性审计产物

结论：`BLOCKED_BY_LABEL_AUDIT`。1,151 条 OMI 中有 235 条不能从当前公开字段唯一恢复 LM/LAD、LCX 或 RCA 责任血管，因此没有实现或训练 Joint-10。

本目录只保存聚合统计、规则、哈希和聚合图，可安全进入 Git。以下含 record/patient 标识的明细文件仅保留在受控服务器，未同步到本地、未提交 Git：

- `unresolved_omi.csv`
- `multigroup_omi.csv`
- `cto_cases.csv`

完整分析见 `docs/ECG_LFM_Nature_Joint10_CV_Comparison_Report.md`。
审计命令、输入/输出哈希见 `run_manifest.json`；独立 wrapper 观察到的真实退出码见 `audit_execution_receipt.json`。
