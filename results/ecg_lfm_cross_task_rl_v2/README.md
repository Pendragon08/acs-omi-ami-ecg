# Cross-task Clinical RL v2 结果索引

本目录保存脱敏后的候选选择、训练轨迹、KL/ranking/cross-reward 历史、inner-validation 快照和曲线图。

* `head_only_selection/candidates.csv`：四组正式 v2A 候选的 Safety、utility 和资源汇总。
* `head_only_selection/selection.json`：Safety-first 选择规则和冻结候选索引。
* `R2A_cross_task_head_*/fold0/`：各候选的配置、训练摘要、KL/ranking/cross-reward 轨迹、inner-validation 快照和图。

checkpoint、原始 ECG、patient-level prediction CSV、manifest 和完整 raw JSONL 不进入 GitHub；受控服务器上保留完整训练材料。最终中文分析见 [`docs/ECG_LFM_CrossTask_RL_v2_report.md`](../../docs/ECG_LFM_CrossTask_RL_v2_report.md)。
