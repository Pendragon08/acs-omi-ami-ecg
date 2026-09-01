# ECG-LFM 多目标梯度方法来源记录

本轮只改变联合训练时共享 LoRA 参数的梯度聚合方式；ACS 的无权重交叉熵、OMI 的无权重 `BCEWithLogitsLoss`、数据划分、模型、LoRA、优化器和调度器保持 W0 不变。

实现来自下列公开 GitHub 仓库的对应算法接口，执行服务器使用的提交如下（MIT License）：

| 方法 | 官方仓库 | 执行时参考提交 | 许可 |
|---|---|---|---|
| PCGrad | [tianheyu927/PCGrad](https://github.com/tianheyu927/PCGrad) | `c5fbd7c856526373828074f06875230f7f3ee79e` | MIT |
| CAGrad | [Cranial-XIX/CAGrad](https://github.com/Cranial-XIX/CAGrad) | `dc3d48152b6196945cfd56144879b9d42353b095` | MIT |
| FairGrad | [OptMN-Lab/FairGrad](https://github.com/OptMN-Lab/FairGrad) | `b58bce810e57288f7bf2c4c9effb7133b856739a` | MIT |
| Aligned-MTL | [taohong08/MTL](https://github.com/taohong08/MTL) | `3f577f1365101fdd16094313b4a3b0d3845ce10d` | MIT |

## 接口适配边界

- PCGrad：按官方 random task order，对共享梯度的冲突项进行投影。
- CAGrad：使用公开实现的二任务 Gram 矩阵和约束求解器，`c=0.5`，任务先验为 `0.5/0.5`。
- FairGrad：使用公开 proportional-fairness 方程，`alpha=1.0`，不额外 clip 或叠加类别权重。
- Aligned-MTL：使用公开 Procrustes/eigendecomposition 求解，`scale_mode=min`，初始任务先验为 `0.5/0.5`。

仅做了 ECG-LFM 接口适配：将两个任务的 autograd 梯度展平为共享向量，再把合成向量写回 LoRA 参数；`proj.weight`/`proj.bias` 的前四行（ACS）和最后一行（OMI）保持 task-specific，不经过梯度 surgery。Fairseq 原有的梯度累积、FP16 loss-scale、all-reduce、gradient clipping 和 Adam step 继续执行。

如果官方求解器依赖缺失导致回退，训练日志会写入 `moo_solver_fallback=1`，不得静默当作完整算法结果。

