# R1-v2 结果索引

本目录保存 R1-v2 Head-only 稳定化研究的可审计结果。实验只使用 nested Fold0 的 inner-train/inner-validation，outer-test 未加载。

- `head_only_selection/candidates.csv`：四组候选汇总。
- `head_only_selection/selection.json`：Safety PASS→clinical utility 选择结果。
- `R1_v2_head_only_*/fold0/`：每组的配置、provenance、训练日志、25/50/75/100 update 验证快照和曲线。
- `train_log.jsonl`：逐 update 日志（为保留完整实验记录，提交时对该文件使用 force-add）。
- `checkpoint_best.pt`：仅保存在服务器，受仓库规则忽略，不上传 Git。

正式候选：

| 候选 | 学习率 | target-KL | 状态 |
|---|---:|---:|---|
| H1 | 1e-6 | 0.005 | completed / Safety PASS |
| H2 | 1e-6 | 0.010 | completed / Safety PASS |
| H3 | 3e-6 | 0.005 | completed / Safety PASS / selected |
| H4 | 3e-6 | 0.010 | completed / Safety PASS |

W0 reference 与 R1-v2 报告见 [`docs/ECG_LFM_R1_v2_stabilization_report.md`](../../docs/ECG_LFM_R1_v2_stabilization_report.md)。
