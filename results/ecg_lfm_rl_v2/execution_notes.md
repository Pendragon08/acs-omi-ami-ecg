# R1-v2 执行记录

| 时间（服务器 UTC） | 事件 | 处理 |
|---|---|---|
| 2026-09-01 | 初次 Head-only 启动审计发现 loader 恢复了 W0 的 LoRA `requires_grad` | 终止未产生结果的进程；runner 增加显式全冻结并只放开 `proj.weight`/`proj.bias`，16 项测试通过 |
| 2026-09-01 | `R1_v2_head_only_probe2` | GPU1 单候选冒烟，100 updates、4 次 inner-validation、Safety PASS；归档，不进入正式选择 |
| 2026-09-01 | H1/H3 | GPU0/GPU1 并行完成，100 updates、Safety PASS |
| 2026-09-01 | H2/H4 | 两条 lane 从 W0 重新初始化后完成，100 updates、Safety PASS |
| 2026-09-01 | 汇总 | `selection.json` 选中 H3（H4 指标并列）；outer-test 未加载 |

正式结果和曲线保存在同级四个 `R1_v2_head_only_*` 目录。启动日志中的 ECG-LFM saved config 只用于服务器审计，不上传仓库；逐 update `train_log.jsonl` 为脱敏训练统计，按仓库记录策略保留。
