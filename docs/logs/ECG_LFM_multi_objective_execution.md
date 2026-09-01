# ECG-LFM 多目标优化执行记录

## 2026-09-01

- 按 `ECG_LFM_Multi_Objective_Optimization_Codex_Report.md` 建立独立 MOO 配置，不改动 W0 的 CE+BCE、LoRA、数据划分、优化器、scheduler、batch 或 checkpoint criterion。
- 新增 `acs_joint_moo` criterion；ACS/OMI 头部行保持 task-specific，只有 trainable LoRA 参数进入共享梯度聚合。
- 新增 PCGrad、CAGrad (`c=0.5`)、FairGrad (`alpha=1.0`)、Aligned-MTL 和 linear-sum 五种方法。
- 远端环境运行 7 项依赖轻量单元测试：全部通过。
- GPU1 完成 1-update 启动烟测：模型构建、FP16 loss-scale、梯度累积、剪裁和 checkpoint 保存均通过。
- Stage A 于 2026-09-01 10:47 UTC 在 GPU1 的 screen `moo_stage_a` 中启动，并于 2026-09-01 21:42 UTC 完成；五种方法均为 Fold0 训练 RC=0、inner-validation 评估 RC=0，outer-test 未访问。
- 训练/评估完成顺序：`linear_sum`（12:56/12:57）、`pcgrad`（15:07/15:08）、`cagrad`（17:19/17:20）、`fairgrad`（19:30/19:31）、`aligned_mtl`（21:41/21:42）。
- Stage A gate 结果：五种方法全部 FAIL，未启动 Stage B nested five-fold；按预注册规则本轮 MOO 结论为 NO-GO。
- Fold0 inner-validation 主指标（ACS Macro AUPRC / OMI AUPRC）：`linear_sum` 0.557115 / 0.451184，`pcgrad` 0.555350 / 0.449083，`cagrad` 0.554291 / 0.448739，`fairgrad` 0.546761 / 0.440173，`aligned_mtl` 0.552545 / 0.449032。
- 汇总脚本默认 W0 路径修正为 `results/multi_objective_evaluation/w0/fold0/inner_metrics.json`，避免遗漏 baseline delta/gate；修正后重新生成 Stage A 表格和中文报告。

## 运行约束

- 服务器：`nlp03`，`CUDA_VISIBLE_DEVICES=1`，单卡 RTX A6000。
- Stage A 日志：服务器 `results/multi_objective/logs/`。
- Stage A 输出：服务器 `results/multi_objective/<method>/fold0/`。
- 训练 raw ECG、patient-level assignment 和 checkpoint 不进入 Git；最终只同步去标识化摘要、指标、曲线和 provenance。

## 完成状态

1. linear-sum、PCGrad、CAGrad、FairGrad、Aligned-MTL 的 Fold0 训练和 inner-validation 评估已全部完成。
2. Stage A gate 汇总已生成：`results/multi_objective_stage_a/stage_a_metrics.csv`、`stage_a_summary.json`、`stage_a_summary.md`，并生成梯度冲突/余弦/范数/有效任务系数/验证 AUPRC 曲线。
3. 五种方法均未通过 gate，因此没有读取 outer-test，也没有启动完整 nested five-fold；不存在可报告的 MOO outer-fold 或 held-out external test 指标。
4. 中文报告：`docs/ECG_LFM_multi_objective_optimization_report.md`。
