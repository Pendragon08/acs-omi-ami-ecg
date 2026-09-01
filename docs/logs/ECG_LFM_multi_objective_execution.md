# ECG-LFM 多目标优化执行记录

## 2026-09-01

- 按 `ECG_LFM_Multi_Objective_Optimization_Codex_Report.md` 建立独立 MOO 配置，不改动 W0 的 CE+BCE、LoRA、数据划分、优化器、scheduler、batch 或 checkpoint criterion。
- 新增 `acs_joint_moo` criterion；ACS/OMI 头部行保持 task-specific，只有 trainable LoRA 参数进入共享梯度聚合。
- 新增 PCGrad、CAGrad (`c=0.5`)、FairGrad (`alpha=1.0`)、Aligned-MTL 和 linear-sum 五种方法。
- 远端环境运行 7 项依赖轻量单元测试：全部通过。
- GPU1 完成 1-update 启动烟测：模型构建、FP16 loss-scale、梯度累积、剪裁和 checkpoint 保存均通过。
- Stage A 已在 GPU1 的 screen `moo_stage_a` 中启动，固定 nested Fold0；当前只写入 inner-validation 结果，outer-test 未访问。

## 运行约束

- 服务器：`nlp03`，`CUDA_VISIBLE_DEVICES=1`，单卡 RTX A6000。
- Stage A 日志：服务器 `results/multi_objective/logs/`。
- Stage A 输出：服务器 `results/multi_objective/<method>/fold0/`。
- 训练 raw ECG、patient-level assignment 和 checkpoint 不进入 Git；最终只同步去标识化摘要、指标、曲线和 provenance。

## 待完成

1. 等待 linear-sum、PCGrad、CAGrad、FairGrad、Aligned-MTL 的 Fold0 训练和 inner-validation 评估。
2. 运行 Stage A gate 汇总；只有通过 gate 的 1–2 个候选允许进入 nested five-fold。
3. 如进入 Stage B，冻结方法后才读取 outer-test，并生成五折统计与最终中文报告。

