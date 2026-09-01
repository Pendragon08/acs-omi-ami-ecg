# ECG-LFM 双任务多目标优化（MOO）Codex 执行报告

> **目标**：在暂停 RL 路线后，基于当前 ECG-LFM + LoRA 联合任务最佳方案（W0），引入**多目标优化 / 多任务梯度优化**，改善 ACS 四分类与 OMI 二分类之间的负迁移或梯度冲突。  
> **核心约束**：**不更换现有任务损失函数，不引入 RL reward，不把两个任务拆成独立模型。**
>
> 训练仍然保留：
>
> \[
> L_{\text{ACS}}=\mathrm{CE}(y_{\text{ACS}}, z_{\text{ACS}})
> \]
>
> \[
> L_{\text{OMI}}=\mathrm{BCEWithLogits}(y_{\text{OMI}}, z_{\text{OMI}})
> \]
>
> \[
> L_{\text{sum}}=L_{\text{ACS}}+L_{\text{OMI}}
> \]
>
> 变化只发生在**共享可训练参数的反向传播/梯度合成方式**上。

---

## 0. 给 Codex 的最终执行结论

不要继续扩展 RL。

本轮只做 **Multi-Objective Optimization / Multi-Task Gradient Optimization**。

### 第一优先级

1. **PCGrad**
2. **CAGrad**
3. **FairGrad**
4. **Aligned-MTL**

### 条件性第二优先级

5. **SAMO**
6. **Nash-MTL**
7. **MGDA**（仅作为经典 Pareto 基线，不作为主推方案）

### 暂不优先

- GradNorm
- Uncertainty Weighting
- DWA
- FAMO
- 继续设计新的动态 loss weighting

原因：上述方法主要还是围绕**损失权重动态调整**。当前研究重点应从“怎么给两个 loss 加权”转向“当两个任务的共享梯度发生冲突时，怎么找到更好的联合更新方向”。

---

# 1. 当前实验基线必须保持不变

Codex 开始改代码前，必须先扫描当前仓库并定位：

- 训练入口；
- ECG-LFM encoder；
- LoRA 注入位置；
- ACS head；
- OMI head；
- loss 计算；
- optimizer / scheduler；
- nested CV；
- inner-validation checkpoint；
- OMI threshold selection；
- evaluation；
- calibration；
- experiment config；
- 当前 W0 结果输出位置。

**不要根据本报告猜文件路径。必须以当前仓库真实结构为准。**

当前 W0 的关键设定必须原样保留：

| 项目 | W0 |
|---|---|
| Encoder | ECG-LFM Foundation Model |
| pooled feature | 1024 |
| 微调方式 | LoRA |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| optimizer LR | 1e-4 |
| ACS loss | unweighted CrossEntropyLoss |
| OMI loss | unweighted BCEWithLogitsLoss |
| joint scalar loss | `L_ACS + L_OMI` |
| ACS inference | 4-class softmax + argmax |
| OMI inference | sigmoid + inner-val threshold |
| OMI threshold | inner-validation Youden |
| checkpoint | inner-validation loss |
| split | patient-level nested CV |
| outer test | freeze 后一次评估 |

### Fold0 当前参考值

用于防止实现后出现回归：

#### ACS

- Macro AUROC: `0.783155`
- Macro AUPRC: `0.546685`
- Macro F1: `0.525076`
- Balanced Accuracy: `0.516938`
- Accuracy: `0.587077`

NSTEMI：

- AUPRC: `0.287459`
- F1: `0.306494`
- Sensitivity: `0.240816`

NSTEMI vs UA：

- AUPRC: `0.526039`

#### OMI

- AUROC: `0.897364`
- AUPRC: `0.412660`
- F1: `0.367113`
- Sensitivity: `0.831169`
- Specificity: `0.815407`
- PPV: `0.235583`
- NPV: `0.986027`
- MCC: `0.378533`
- Accuracy: `0.816417`

Inner-val Youden threshold：

- `0.04`

NSTE-ACS 中 OMI：

- AUPRC: `0.227752`

> 注意：这些数值只能作为当前 Fold0 的工程回归参考。最终是否“涨点”必须依赖完整 nested 5-fold，而不是只看 Fold0。

---

# 2. 为什么这次应做多目标优化，而不是继续调 loss 权重

当前联合训练形式本身没有问题：

\[
L=L_{\text{ACS}}+L_{\text{OMI}}
\]

真正可能的问题发生在共享参数上。

设 ECG-LFM/LoRA 的共享可训练参数为：

\[
\theta_s
\]

两个任务分别产生：

\[
g_{\text{ACS}}
=
\nabla_{\theta_s} L_{\text{ACS}}
\]

\[
g_{\text{OMI}}
=
\nabla_{\theta_s} L_{\text{OMI}}
\]

普通 W0 实际使用：

\[
g_{\text{W0}}
=
g_{\text{ACS}}
+
g_{\text{OMI}}
\]

如果：

\[
g_{\text{ACS}}^\top g_{\text{OMI}} < 0
\]

说明两个任务在当前更新步骤上存在**梯度冲突**。

此时简单相加会导致：

- ACS 更新方向损害 OMI；
- 或 OMI 更新方向损害 ACS；
- 或某个梯度范数长期压制另一个任务；
- 出现 negative transfer；
- 两个任务联合训练，却没有真正共享到有益的信息。

因此本轮不要再优先研究：

\[
w_1 L_{\text{ACS}}+w_2 L_{\text{OMI}}
\]

而应研究：

\[
g_{\text{joint}}
=
\mathcal A
(
g_{\text{ACS}},
g_{\text{OMI}}
)
\]

其中 \(\mathcal A\) 是多目标梯度聚合器。

---

# 3. 与本项目最相关的顶会参考文献

## 3.1 MGDA / Multi-Task Learning as Multi-Objective Optimization

**Sener, O., & Koltun, V.**  
**Multi-Task Learning as Multi-Objective Optimization**  
**NeurIPS 2018**

论文：

https://proceedings.neurips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html

核心思想：

把 MTL 明确建模为：

\[
\min_{\theta}
\left(
L_1(\theta),
L_2(\theta),
\dots
\right)
\]

而不是只把多个任务当成固定加权和。

通过 Multiple Gradient Descent Algorithm（MGDA）寻找 Pareto-stationary 更新方向。

### 对本项目意义

这是本轮“ACS + OMI 是两个优化目标，而不是一个 RL reward”的理论基础。

### 本轮定位

经典基线，优先级低于 CAGrad / FairGrad / Aligned-MTL。

---

## 3.2 PCGrad

**Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C.**  
**Gradient Surgery for Multi-Task Learning**  
**NeurIPS 2020**

论文：

https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html

官方代码（TensorFlow）：

https://github.com/tianheyu927/PCGrad

常用 PyTorch 实现：

https://github.com/WeiChengTseng/Pytorch-PCGrad

### 核心思想

如果：

\[
g_i^\top g_j < 0
\]

就把冲突部分从梯度中投影掉。

例如：

\[
g_i'
=
g_i-
\frac{g_i^\top g_j}
{\|g_j\|^2}
g_j
\]

最终再聚合修正后的梯度。

### 对本项目优势

- 不改 CE；
- 不改 BCE；
- 不需要 reward；
- 不需要训练额外网络；
- 没有复杂超参数；
- 两任务时实现尤其简单；
- 非常适合作为第一条 MOO baseline。

### 本轮优先级

**高。**

---

# 3.3 CAGrad

**Liu, B., Liu, X., Jin, X., Stone, P., & Liu, Q.**  
**Conflict-Averse Gradient Descent for Multi-task Learning**  
**NeurIPS 2021**

论文：

https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html

官方代码：

https://github.com/Cranial-XIX/CAGrad

### 核心思想

CAGrad 不仅避免梯度冲突，还试图：

- 保持对平均任务损失的优化；
- 同时改善最不利任务的局部下降方向；
- 在总体性能和 worst-task improvement 之间做平衡。

论文明确把 ordinary average-loss gradient descent 与多目标优化联系起来。

### 为什么最适合当前 W0

当前：

\[
L_{\text{sum}}
=
L_{\text{ACS}}
+
L_{\text{OMI}}
\]

对于两个任务：

\[
\frac{1}{2}
\left(
L_{\text{ACS}}+
L_{\text{OMI}}
\right)
\]

与求和仅相差常数缩放，因此 CAGrad 非常符合“**loss 还是相加，但更新方向更合理**”这一要求。

### 初始超参数

第一轮只使用：

```text
c = 0.5
```

不要一开始做大规模调参。

只有当 Fold0 明确优于 W0 时，再考虑：

```text
c ∈ {0.2, 0.5, 0.8}
```

### 本轮优先级

**最高之一。**

---

# 3.4 Nash-MTL

**Navon, A., Shamsian, A., Achituve, I., Maron, H., Kawaguchi, K., Chechik, G., & Fetaya, E.**  
**Multi-Task Learning as a Bargaining Game**  
**ICML 2022**

论文：

https://proceedings.mlr.press/v162/navon22a.html

官方代码：

https://github.com/AvivNavon/nash-mtl

### 核心思想

把两个任务看作两个参与博弈的目标。

它们对联合参数更新方向进行 Nash bargaining，而不是人工规定某一个任务权重必须是多少。

### 优点

适合：

- 两个目标的重要性都高；
- 不希望一个任务通过梯度尺度“压制”另一个任务；
- 希望获得较为 principled 的联合梯度。

### 缺点

- 实现比 PCGrad / CAGrad 复杂；
- 每一步有额外求解开销；
- 本项目当前只有两个任务，没必要第一轮就用最复杂方案。

### 本轮优先级

**第二阶段。**

---

# 3.5 Aligned-MTL

**Senushkin, D., Patakin, N., Kuznetsov, A., & Konushin, A.**  
**Independent Component Alignment for Multi-Task Learning**  
**CVPR 2023**

论文：

https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html

官方 PyTorch：

https://github.com/taohong08/MTL

### 核心思想

把不同任务的梯度组成一个梯度系统，通过其 condition number 判断优化稳定性，并对梯度独立成分进行对齐。

它特别针对：

- conflicting gradients；
- dominating gradients；
- multi-task training instability。

### 对本项目优势

本项目原始先验就是：

```text
ACS : OMI = 1 : 1
```

Aligned-MTL 支持预定义 task trade-off，因此可以继续保持两个任务等重要，而不是重新搜索 loss 权重。

### 本轮优先级

**高。**

---

# 3.6 FairGrad

**Ban, H., & Ji, K.**  
**Fair Resource Allocation in Multi-Task Learning**  
**ICML 2024**

论文：

https://proceedings.mlr.press/v235/ban24a.html

官方代码：

https://github.com/OptMN-Lab/FairGrad

### 核心思想

把多任务优化视为不同任务之间的公平资源分配。

关注：

\[
\Delta L_{\text{ACS}}
\]

与：

\[
\Delta L_{\text{OMI}}
\]

是否被公平优化，而不是只优化平均效果。

论文在多任务梯度方法上报告了较强结果，并提供理论收敛分析。

### 为什么适合 ACS + OMI

ACS 和 OMI：

- label space 不同；
- loss 类型不同；
- 梯度尺度可能不同；
- 临床意义都不能被另一个任务完全覆盖。

FairGrad 的“不要让某一个任务长期欠优化”的思想，与当前联合任务非常匹配。

### 第一轮配置

优先测试：

```text
alpha = 1.0
```

先使用论文/官方实现对应的 proportional-fairness 形式。

不要第一轮搜索大量 alpha。

### 本轮优先级

**高。**

---

# 3.7 SAMO

**Ban, H., Subramani, G. R., & Ji, K.**  
**SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation**  
**ICCV 2025**

论文：

https://openaccess.thecvf.com/content/ICCV2025/html/Ban_SAMO_A_Lightweight_Sharpness-Aware_Approach_for_Multi-Task_Optimization_with_Joint_ICCV_2025_paper.html

官方代码：

https://github.com/OptMN-Lab/SAMO

### 核心思想

将 Sharpness-Aware Optimization 引入多任务学习，同时利用：

- global average-loss information；
- task-local information。

目标之一是减轻：

- task conflict；
- sharp minima；
- negative transfer。

### 为什么值得作为第二阶段

这是较新的顶会方法。

而当前只训练 LoRA + heads，可训练参数相对少，因此 SAM 类额外计算在本项目中可能比 full fine-tuning 更可接受。

### 本轮定位

如果 PCGrad / CAGrad / FairGrad / Aligned-MTL 没有带来稳定提升，再测试 SAMO。

不要一开始就把工程复杂度拉高。

---

# 3.8 FAMO：只列为参考，不作为本轮主线

**Liu, B., Feng, Y., Stone, P., & Liu, Q.**  
**FAMO: Fast Adaptive Multitask Optimization**  
**NeurIPS 2023**

论文：

https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract.html

官方代码：

https://github.com/Cranial-XIX/FAMO

FAMO 是优秀顶会方法，但核心更接近动态优化任务下降速率 / 动态权重。

考虑到本项目已经花过较多实验预算研究 loss weighting，本轮不优先重复这一方向。

---

# 4. 本轮推荐实验矩阵

## Stage A：Fold0 快速筛选

必须使用：

- 同一个 Fold0；
- 同一个 patient split；
- 同一个 seed；
- 同一个训练 epoch；
- 同一个 early stopping；
- 同一个 LR；
- 同一个 batch size；
- 同一个 LoRA；
- 同一个 threshold protocol；
- 同一个 calibration protocol；
- 同一个数据预处理。

仅改变：

```text
shared-gradient aggregation method
```

实验：

| ID | 方法 | Loss | Shared gradient |
|---|---|---|---|
| MOO-0 | W0 | CE + BCE | direct sum |
| MOO-1 | PCGrad | CE + BCE | PCGrad |
| MOO-2 | CAGrad | CE + BCE | CAGrad, c=0.5 |
| MOO-3 | FairGrad | CE + BCE | FairGrad, alpha=1.0 |
| MOO-4 | Aligned-MTL | CE + BCE | Aligned-MTL, equal task prior |

### 不要在 Stage A 做

- 多 seed；
- 全部 5 folds；
- CAGrad 大网格；
- FairGrad 大网格；
- Nash-MTL；
- SAMO；
- 新 loss；
- 新数据增强；
- 新分类头；
- 新 LoRA rank；
- 新 threshold 方法。

目的只有一个：

> 判断“共享梯度冲突优化”有没有明显信号。

---

# 5. 如果 Stage A 有涨点，再进入 Stage B

候选方法必须满足：

### Fold0 screening gate

令：

\[
\Delta A
=
AUPRC_{\text{ACS,macro}}^{candidate}
-
AUPRC_{\text{ACS,macro}}^{W0}
\]

\[
\Delta O
=
AUPRC_{\text{OMI}}^{candidate}
-
AUPRC_{\text{OMI}}^{W0}
\]

满足至少一种：

### 条件 1

```text
min(ΔA, ΔO) >= -0.003
and
max(ΔA, ΔO) >= +0.005
```

即：

- 一个任务至少有可见提升；
- 另一个任务不能明显掉点。

### 条件 2

```text
ΔA >= +0.003
and
ΔO >= +0.003
```

即：

- 两个任务同时小幅提升。

同时检查：

```text
NSTEMI vs UA AUPRC
NSTE-ACS OMI AUPRC
ACS Macro F1
OMI MCC
OMI sensitivity
OMI specificity
```

如果 primary AUPRC 有提升，但 NSTE 关键指标明显恶化，也不能直接判定 GO。

---

# 6. Stage B：完整 nested 5-fold

只推广 Stage A 中最好的 **1–2 个**方法。

例如：

```text
W0
CAGrad
FairGrad
```

或：

```text
W0
PCGrad
Aligned-MTL
```

不要把 7 个方法全部跑完整 5-fold。

Stage B 输出：

```text
mean ± std
95% CI
fold-wise metrics
paired differences vs W0
```

必须至少报告：

### ACS

```text
Macro AUROC
Macro AUPRC
Macro F1
Balanced Accuracy
Accuracy
per-class AUPRC
per-class F1
NSTEMI sensitivity
NSTEMI vs UA AUPRC
```

### OMI

```text
AUROC
AUPRC
F1
Sensitivity
Specificity
PPV
NPV
MCC
Accuracy
NSTE-ACS OMI AUPRC
```

### Calibration

保持原校准流程，继续报告：

```text
Brier
ECE
```

---

# 7. 最终方法选择必须是“多目标”而不是只看一个数字

不要因为 ACS 涨 0.02、OMI 掉 0.03，就宣称成功。

增加以下工程比较指标。

## 7.1 Relative task gain

\[
r_A
=
\frac{
M_A^{candidate}-M_A^{W0}
}{
M_A^{W0}
}
\]

\[
r_O
=
\frac{
M_O^{candidate}-M_O^{W0}
}{
M_O^{W0}
}
\]

其中默认：

```text
M_A = ACS Macro AUPRC
M_O = OMI AUPRC
```

---

## 7.2 Balanced Gain

\[
G_{\text{balanced}}
=
\min(r_A,r_O)
\]

该指标体现：

> 最差的那个任务到底有没有得到改善。

---

## 7.3 Geometric Joint Score

\[
S_{\text{joint}}
=
\sqrt{
AUPRC_{\text{ACS,macro}}
\cdot
AUPRC_{\text{OMI}}
}
\]

只能作为辅助 summary score。

**不能用它替代两个任务的单独指标。**

---

# 8. 共享参数和 task-specific 参数必须分开处理

这是本实现最关键的地方。

模型逻辑：

```text
ECG input
    │
    ▼
ECG-LFM
    │
LoRA adapters
    │
    ▼
shared representation
    ├─────────────► ACS head ─► 4 logits ─► CE
    │
    └─────────────► OMI head ─► 1 logit  ─► BCE
```

定义：

```text
shared_params
task_acs_params
task_omi_params
```

### shared_params

包含所有同时受到 ACS 和 OMI supervision 的**可训练参数**，例如：

- ECG-LFM LoRA adapters；
- 如果存在共享 projection / pooling adapter，则包括它。

### 不包含

- frozen ECG-LFM backbone；
- ACS head；
- OMI head。

### ACS head

只接受：

\[
\nabla_{\theta_{\text{ACS}}}L_{\text{ACS}}
\]

### OMI head

只接受：

\[
\nabla_{\theta_{\text{OMI}}}L_{\text{OMI}}
\]

### shared LoRA

才使用：

\[
\mathcal A(g_{\text{ACS}},g_{\text{OMI}})
\]

这是 MOO 实现的核心。

---

# 9. 不允许错误地对两个 head 做梯度 surgery

错误：

```text
把整个 model.parameters() flatten
然后直接对两个 loss 做 PCGrad/CAGrad
```

因为：

- ACS head 本来就没有 OMI gradient；
- OMI head 本来就没有 ACS gradient；
- task-specific 参数不需要人为做两个任务的冲突消解。

正确：

```text
shared params:
    MOO aggregation

ACS head:
    ACS gradient

OMI head:
    OMI gradient
```

---

# 10. 必须新增梯度冲突诊断

不要额外开一组只做诊断的实验。

在所有 Stage A 训练中直接记录即可。

每个 training step 或每 N steps 记录：

## 10.1 cosine similarity

\[
\cos(g_A,g_O)
=
\frac{
g_A^\top g_O
}{
\|g_A\|\|g_O\|+\epsilon
}
\]

日志字段：

```text
grad_cosine_acs_omi
```

---

## 10.2 conflict flag

```text
grad_conflict = grad_cosine_acs_omi < 0
```

epoch 汇总：

```text
gradient_conflict_rate
```

即：

\[
R_{\text{conflict}}
=
\frac{
\#\{g_A^\top g_O<0\}
}{
N_{\text{steps}}
}
\]

---

## 10.3 gradient norm

记录：

```text
grad_norm_acs_shared
grad_norm_omi_shared
grad_norm_ratio_acs_over_omi
```

---

## 10.4 post-aggregation gradient

记录：

```text
grad_norm_joint
```

若方法支持，还记录：

```text
effective_task_weight_acs
effective_task_weight_omi
```

例如：

- MGDA；
- CAGrad；
- FairGrad；
- Nash-MTL。

---

# 11. PCGrad 实现要求

设：

```python
g_acs
g_omi
```

为共享参数 flatten 后梯度。

若：

```python
dot(g_acs, g_omi) < 0
```

进行 projection。

两任务场景可以使用论文逻辑：

\[
g_A'
=
g_A-
\frac{
g_A^\top g_O
}{
\|g_O\|^2+\epsilon
}
g_O
\]

\[
g_O'
=
g_O-
\frac{
g_O^\top g_A
}{
\|g_A\|^2+\epsilon
}
g_A
\]

然后：

\[
g_{\text{PCGrad}}
=
\frac{g_A'+g_O'}{2}
\]

或严格按论文/成熟 PyTorch 实现的 random task order 方式执行。

### 要求

优先参考成熟实现，不要自己写一个与原论文语义不同的“简化版 PCGrad”。

---

# 12. CAGrad 实现要求

优先参考：

https://github.com/Cranial-XIX/CAGrad

不要重写算法数学求解器。

第一轮：

```yaml
method: cagrad
c: 0.5
task_prior:
  acs: 0.5
  omi: 0.5
```

最终共享参数更新必须来自：

```text
CAGrad(g_acs, g_omi)
```

task heads 不经过 CAGrad。

---

# 13. FairGrad 实现要求

优先参考：

https://github.com/OptMN-Lab/FairGrad

第一轮：

```yaml
method: fairgrad
alpha: 1.0
```

不要自行修改论文中的：

- solver；
- gradient Gram matrix；
- fairness equation；
- effective gradient weight。

只做本项目接口适配。

记录：

```text
fairgrad_weight_acs
fairgrad_weight_omi
```

并监控是否出现：

```text
NaN
Inf
extreme weight
```

如果权重出现严重数值异常：

- 不要偷偷 clip 改算法；
- 先把异常写入报告；
- 再参考官方实现的数值策略修复。

---

# 14. Aligned-MTL 实现要求

参考：

https://github.com/taohong08/MTL

第一轮：

```yaml
method: aligned_mtl
task_weights:
  acs: 0.5
  omi: 0.5
```

重点保持：

```text
1:1 task prior
```

不要在第一轮改为：

```text
0.7 : 0.3
0.3 : 0.7
```

因为本轮目的是比较**优化器本身**，不是重新做 loss weight search。

---

# 15. SAMO 只在 Stage A 全部失败后运行

触发条件：

```text
PCGrad
CAGrad
FairGrad
Aligned-MTL
```

均不能满足 Stage A screening gate。

再实现：

```text
SAMO
```

参考：

https://github.com/OptMN-Lab/SAMO

原因：

- 方法新；
- ICCV 2025；
- sharpness-aware 有可能改善泛化；
- 但训练逻辑复杂度高于前四者。

本轮禁止一开始就把 SAMO 与前四种全部同时全量实现。

---

# 16. Nash-MTL 只作为进一步 Pareto 验证

若发现：

```text
ACS 与 OMI 梯度冲突率高
```

且：

```text
PCGrad / CAGrad 有部分效果
```

但两个任务仍明显此消彼长，则再跑 Nash-MTL。

参考：

https://github.com/AvivNavon/nash-mtl

它适合验证：

> 更 principled 的 bargaining solution 是否优于 projection / conflict-averse 更新。

---

# 17. AMP / mixed precision 必须特别检查

如果当前训练使用：

```text
torch.cuda.amp
autocast
GradScaler
```

必须验证：

```text
task gradient extraction
gradient dot product
gradient norm
projection
gradient aggregation
```

发生在数值正确的尺度上。

禁止：

- 一个任务 gradient scaled、另一个没 scaled；
- 对 scaled gradient 做 norm clipping 后直接赋值；
- 手工 `.grad` 写入后又被 GradScaler 二次处理；
- 因 flatten/unflatten 导致 dtype 改变。

增加数值单测：

```text
AMP off vs AMP on
```

在一个小 batch 上检查：

```text
cosine sign approximately identical
no NaN
no Inf
parameter update finite
```

---

# 18. Gradient accumulation 必须保持原行为

如果当前：

```text
gradient_accumulation_steps > 1
```

不能为了 MOO 偷偷变成：

```text
accumulation_steps = 1
```

推荐：

每个 micro-batch：

1. 计算 `L_ACS`；
2. 计算 `L_OMI`；
3. 获取两个 task 的 shared gradient；
4. 执行 MOO；
5. 把 merged shared gradient 累加；
6. task head gradient 正常累加；
7. 到 accumulation boundary 再 `optimizer.step()`。

确保 effective batch size 与 W0 一致。

---

# 19. Gradient clipping 必须放在正确位置

如果原训练存在：

```python
clip_grad_norm_
```

正确顺序：

```text
per-task gradient calculation
    ↓
MOO aggregation
    ↓
write merged gradient to .grad
    ↓
global gradient clipping
    ↓
optimizer.step()
```

不要先分别把两个 task gradient clip 到同样大小再做 PCGrad/CAGrad，除非原论文算法明确如此。

---

# 20. Scheduler / optimizer 不允许跟着改变

MOO 不是重新设计 optimizer。

保持：

```text
AdamW / 当前 optimizer
lr = 1e-4
scheduler = 当前 W0
weight decay = 当前 W0
warmup = 当前 W0
```

MOO 是 optimizer step 前的 gradient aggregation layer。

建议结构：

```python
losses = {
    "acs": loss_acs,
    "omi": loss_omi,
}

moo_result = moo_controller.backward(
    losses=losses,
    shared_params=shared_params,
    task_params={
        "acs": acs_head_params,
        "omi": omi_head_params,
    },
)

optimizer.step()
```

不要强制完全照这个 API 名字实现，但代码职责必须类似。

---

# 21. 推荐代码架构

Codex 应优先实现一个独立模块，例如：

```text
multi_objective/
    __init__.py
    base.py
    linear_sum.py
    pcgrad.py
    cagrad.py
    fairgrad.py
    aligned_mtl.py
    diagnostics.py
```

如果当前项目已有 optimizer / training utilities，则遵循现有目录风格。

不要为了符合本报告强制创建完全一样的路径。

统一接口建议：

```python
class MultiObjectiveStrategy:
    def backward(
        self,
        loss_acs,
        loss_omi,
        shared_params,
        acs_params,
        omi_params,
    ):
        ...
```

---

# 22. CLI / config

新增一个配置项：

```yaml
multi_objective:
  method: linear_sum
```

支持：

```text
linear_sum
pcgrad
cagrad
fairgrad
aligned_mtl
```

扩展项：

```text
nash_mtl
samo
mgda
```

CAGrad：

```yaml
multi_objective:
  method: cagrad
  c: 0.5
```

FairGrad：

```yaml
multi_objective:
  method: fairgrad
  alpha: 1.0
```

Aligned-MTL：

```yaml
multi_objective:
  method: aligned_mtl
  task_weights: [0.5, 0.5]
```

---

# 23. 必须保证 linear_sum 完全复现 W0

实现 MOO framework 后首先运行：

```text
method = linear_sum
```

它必须尽可能复现原 W0。

允许训练随机性导致极小差异，但不能出现：

- 数据 split 变化；
- loss 变化；
- threshold 变化；
- checkpoint rule 变化；
- 模型参数量变化；
- optimizer 变化；
- metric 计算变化。

如果 linear_sum 无法复现 W0：

> **停止 MOO 对比，先排查代码回归。**

---

# 24. Checkpoint 规则不要为了 MOO 改掉

当前：

```text
checkpoint = minimum inner-validation joint loss
```

继续使用：

\[
L_{\text{val}}
=
L_{\text{ACS,val}}
+
L_{\text{OMI,val}}
\]

不要在本轮改成：

```text
best ACS AUPRC
best OMI AUPRC
best JointScore
```

否则会同时改变：

- 训练优化；
- 模型选择；

无法判断涨点来自 MOO 还是 checkpoint selection。

---

# 25. OMI threshold 也必须保持原流程

保持：

```text
inner validation
    ↓
Youden threshold
    ↓
freeze threshold
    ↓
outer test once
```

禁止：

```text
outer test 上重新找 threshold
```

---

# 26. ACS 继续禁止阈值化

ACS 是互斥四分类：

```text
OTHER
STEMI
NSTEMI
UA
```

继续：

```python
probs = softmax(logits)
pred = argmax(probs)
```

禁止为四个类别分别调独立阈值。

---

# 27. 校准仍然是 post-hoc

MOO 不应改变当前 calibration protocol。

训练完成后按原方案：

```text
temperature scaling / existing calibration
```

报告：

```text
pre-calibration
post-calibration
```

的：

```text
Brier
ECE
```

---

# 28. 单元测试

必须新增至少以下测试。

## test_01_no_conflict_pcgrad

构造：

```text
g1 · g2 > 0
```

期望：

```text
PCGrad 不做冲突投影
```

---

## test_02_conflict_pcgrad

构造：

```text
g1 · g2 < 0
```

期望：

```text
projection 后冲突项被消除/减弱
```

---

## test_03_shared_vs_head

小 toy network：

```text
shared trunk
ACS head
OMI head
```

检查：

```text
ACS head only receives ACS loss gradient
OMI head only receives OMI loss gradient
shared trunk receives MOO gradient
```

---

## test_04_linear_sum_equivalence

对同一 batch：

旧 W0：

```python
(loss_acs + loss_omi).backward()
```

新框架：

```text
method=linear_sum
```

检查 shared/head gradients 数值近似相同。

---

## test_05_no_nan

所有方法：

```text
PCGrad
CAGrad
FairGrad
Aligned-MTL
```

检查：

```text
grad finite
loss finite
weights finite
optimizer step finite
```

---

## test_06_frozen_encoder

确认：

```text
ECG-LFM frozen backbone
```

仍然：

```text
requires_grad=False
grad is None
```

只有目标 LoRA 和 heads 更新。

---

## test_07_checkpoint_threshold_protocol

确认：

```text
inner-val checkpoint
inner-val OMI threshold
```

没有因为新 MOO 模块发生变化。

---

# 29. 训练日志

每个 epoch 至少输出：

```text
train_loss_acs
train_loss_omi
train_loss_sum

val_loss_acs
val_loss_omi
val_loss_sum

grad_cosine_mean
grad_cosine_median
gradient_conflict_rate

grad_norm_acs_mean
grad_norm_omi_mean
grad_norm_joint_mean
```

如果有动态 gradient weights：

```text
effective_weight_acs_mean
effective_weight_omi_mean
```

---

# 30. 新增 MOO 诊断图

自动输出：

## 图 1

```text
epoch vs gradient_conflict_rate
```

## 图 2

```text
epoch vs mean gradient cosine
```

## 图 3

```text
epoch vs ACS shared gradient norm
epoch vs OMI shared gradient norm
```

## 图 4

```text
ACS validation AUPRC
OMI validation AUPRC
```

## 图 5

如果方法有动态权重：

```text
effective ACS gradient weight
effective OMI gradient weight
```

---

# 31. 结果表

自动生成：

```markdown
| Method | ACS Macro AUPRC | ACS Macro F1 | NSTEMI-vs-UA AUPRC | OMI AUPRC | NSTE OMI AUPRC | OMI MCC | Conflict Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| W0 | ... | ... | ... | ... | ... | ... | ... |
| PCGrad | ... | ... | ... | ... | ... | ... | ... |
| CAGrad | ... | ... | ... | ... | ... | ... | ... |
| FairGrad | ... | ... | ... | ... | ... | ... | ... |
| Aligned-MTL | ... | ... | ... | ... | ... | ... | ... |
```

再自动计算：

```text
Δ vs W0
relative gain
balanced gain
joint geometric score
```

---

# 32. 必须回答“有没有真正解决梯度冲突”

最终报告不能只写：

```text
CAGrad AUPRC +0.01
```

还必须回答：

### 1.

W0 的：

```text
gradient_conflict_rate
```

到底是多少？

### 2.

冲突主要发生在哪个训练阶段？

```text
early
middle
late
```

### 3.

两个任务梯度范数差距多大？

### 4.

MOO 方法是否：

```text
降低冲突率
```

或：

```text
在存在冲突时改变了更新方向
```

### 5.

冲突改善是否真的对应验证/测试指标改善？

否则不能证明：

> MOO 解决了当前联合任务优化问题。

---

# 33. 不允许使用 outer test 指导方法选择

严格禁止：

```text
跑 Fold0 outer test
看哪个方法最好
再调方法超参
再重新跑同一个 outer test
```

正确：

```text
training
    ↓
inner validation
    ↓
method/hyperparameter decision
    ↓
freeze
    ↓
outer test
```

Stage A 可以把 Fold0 作为工程 screening，但必须明确：

> Fold0 只用于筛选值得进入完整 nested 5-fold 的方法，不能被写成最终泛化结论。

---

# 34. 完整 5-fold 后的统计判定

最终推荐方法必须基于：

```text
paired fold-wise comparison
```

优先：

```text
paired bootstrap
```

或已有统计方案。

最终 GO 的核心条件：

### GO

至少满足：

1. ACS Macro AUPRC 对 W0 不劣；
2. OMI AUPRC 对 W0 不劣；
3. 至少一个 primary task 有稳定提升；
4. NSTE 指标没有明显恶化；
5. 结果不是单个 Fold 偶然；
6. calibration 不出现明显恶化。

### STRONG GO

两个 primary task：

```text
ACS Macro AUPRC
OMI AUPRC
```

都稳定提升。

### NO-GO

如果完整 nested 5-fold 后：

```text
一边涨，一边稳定下降
```

或者：

```text
Fold0 涨，但跨 fold 不稳定
```

则停止该 MOO 方法。

---

# 35. 最推荐的论文实验结构

如果最终 CAGrad / FairGrad / Aligned-MTL 中某一个成功，可以在论文中形成：

## Baseline

```text
ECG-LFM + LoRA + Equal-Sum Joint Training
```

## Multi-objective ablation

```text
Equal Sum
PCGrad
CAGrad
FairGrad
Aligned-MTL
```

## 分析

```text
gradient conflict rate
gradient cosine similarity
task gradient norm
Pareto / balanced gain
```

这比：

```text
W0
W1
W2
W3
不同 loss 权重
```

更有方法论意义。

---

# 36. 如果 CAGrad 最好，推荐最终方法命名

例如：

```text
ECG-LFM LoRA Joint-CAGrad
```

论文方法描述可写为：

> We formulate joint ACS subtype classification and OMI detection as a two-objective multi-task optimization problem. The original task losses remain an unweighted cross-entropy loss for ACS classification and an unweighted binary cross-entropy loss for OMI detection. Instead of modifying the task objectives, conflict-averse gradient aggregation is applied to the shared LoRA-adapted ECG representation parameters, while task-specific classification heads retain their native task gradients.

注意：

> 不要声称 CAGrad 是本文原创算法。

创新只能是：

- 将联合 ACS + OMI 明确建模为 two-objective learning；
- 对共享 ECG foundation model LoRA 参数做 task-gradient conflict optimization；
- 保持 task-specific heads 解耦；
- 用临床关键 subgroup 指标验证 negative transfer；
- 做 gradient conflict mechanism analysis。

---

# 37. 如果所有 MOO 方法都没涨点，结论也有价值

如果：

```text
PCGrad
CAGrad
FairGrad
Aligned-MTL
```

全部失败，并且日志显示：

```text
W0 gradient conflict rate 很低
```

则可以得到重要结论：

> 当前 ACS + OMI 的主要瓶颈不是 gradient conflict。

下一步应转向：

- representation limitation；
- NSTEMI / UA feature separability；
- NSTE-ACS 内 OMI 表征；
- label noise；
- subgroup feature learning；
- frequency-domain / morphology expert；
- task-specific adapter；
- partial parameter sharing。

这比继续盲目调 loss 权重更能缩小问题范围。

---

# 38. Codex 实施步骤

## Step 1：仓库审计

输出：

```text
当前训练入口
模型定义
LoRA 参数
loss
optimizer
scheduler
nested CV
checkpoint
threshold
metric
calibration
```

列出真实路径和函数名。

---

## Step 2：实现 shared/task-specific parameter partition

验证：

```text
shared trainable params
ACS-specific params
OMI-specific params
frozen params
```

输出参数数量。

---

## Step 3：新增 MOO abstraction

先实现：

```text
linear_sum
```

保证复现 W0。

---

## Step 4：实现 diagnostics

加入：

```text
cosine
conflict rate
norm
```

---

## Step 5：PCGrad

实现 + 单测。

---

## Step 6：CAGrad

参考官方 repo 实现 + 单测。

---

## Step 7：FairGrad

参考官方 repo 实现 + 单测。

---

## Step 8：Aligned-MTL

参考官方 repo 实现 + 单测。

---

## Step 9：回归测试

运行：

```text
linear_sum
```

确保当前 W0 pipeline 没被破坏。

---

## Step 10：Stage A Fold0

依次运行：

```text
MOO-0 W0
MOO-1 PCGrad
MOO-2 CAGrad
MOO-3 FairGrad
MOO-4 Aligned-MTL
```

---

## Step 11：自动生成 Stage A 报告

必须包含：

```text
metrics
Δ vs W0
gradient conflict
gradient cosine
gradient norm
runtime
GPU memory
```

---

## Step 12：按照 screening gate 自动推荐 0–2 个候选

不能凭人工主观挑方法。

---

## Step 13：只对候选运行 nested 5-fold

---

## Step 14：生成最终报告

文件建议：

```text
ECG_LFM_multi_objective_optimization_report.md
```

同时保存：

```text
results.csv
fold_metrics.csv
gradient_diagnostics.csv
configs/
figures/
```

---

# 39. Codex 最终报告模板

```markdown
# ECG-LFM Multi-Objective Optimization Report

## 1. Repository audit

## 2. Baseline reproduction

## 3. Parameter partition
- shared
- ACS-specific
- OMI-specific
- frozen

## 4. Implemented MOO methods
- PCGrad
- CAGrad
- FairGrad
- Aligned-MTL

## 5. Gradient-conflict diagnostics

## 6. Fold0 screening

## 7. Candidate promotion decision

## 8. Nested 5-fold results

## 9. Calibration

## 10. NSTE subgroup analysis

## 11. Statistical comparison vs W0

## 12. Runtime / GPU memory

## 13. GO / NO-GO

## 14. Recommended final model
```

---

# 40. 禁止事项

Codex 本轮禁止：

- 重写 dataset split；
- 删除任何患者；
- 修改 ACS label；
- 修改 OMI label；
- 改 ACS 为 4 个独立 sigmoid；
- 给 ACS 类别设独立 threshold；
- outer-test threshold tuning；
- class weighted CE；
- class weighted BCE；
- focal loss；
- LDAM；
- RL；
- PPO；
- GRPO；
- reward；
- 改 LoRA rank；
- 改 encoder；
- 改输入采样率；
- 改 preprocessing；
- 同时引入新的数据增强；
- 同时更换分类头；
- 用 outer test 反复调 MOO 超参。

本轮只能回答：

> **在当前 W0 不变的前提下，多目标共享梯度优化能不能改善 ACS + OMI 联合训练？**

---

# 41. 本报告推荐顺序

综合：

- 理论匹配程度；
- 代码成熟度；
- 工程风险；
- 两任务适配性；
- 顶会证据；
- 与“loss 继续相加”的兼容性；

建议顺序：

```text
CAGrad
    ↓
PCGrad
    ↓
FairGrad
    ↓
Aligned-MTL
    ↓
SAMO
    ↓
Nash-MTL
```

其中第一轮实际同时测试：

```text
W0
PCGrad
CAGrad
FairGrad
Aligned-MTL
```

---

# 42. 最终研究判断

当前问题不建议继续解释成：

> “reward 没设计好，所以 RL 没涨点。”

更合理的研究假设是：

> ACS subtype classification 与 OMI detection 在共享 ECG-LFM 表征上可能存在不同程度的梯度冲突、梯度尺度不平衡或 negative transfer。

因此下一条最干净的路线是：

\[
\boxed{
\text{ECG-LFM + LoRA}
+
\text{CE}_{ACS}
+
\text{BCE}_{OMI}
+
\text{Multi-Objective Gradient Optimization}
}
\]

而不是：

\[
\boxed{
\text{ECG-LFM}
+
\text{RL reward engineering}
}
\]

首选验证：

\[
\boxed{
\text{CAGrad / PCGrad / FairGrad / Aligned-MTL}
}
\]

只要其中一种在完整 nested 5-fold 中实现：

```text
ACS 不掉
+
OMI 不掉
+
至少一个稳定涨点
```

就证明：

> 多目标梯度优化比继续调整 joint loss 权重更适合当前双任务。

---

# 参考文献

1. Sener O, Koltun V. **Multi-Task Learning as Multi-Objective Optimization.** NeurIPS 2018.  
   https://proceedings.neurips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html

2. Yu T, Kumar S, Gupta A, Levine S, Hausman K, Finn C. **Gradient Surgery for Multi-Task Learning.** NeurIPS 2020.  
   https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html

3. Liu B, Liu X, Jin X, Stone P, Liu Q. **Conflict-Averse Gradient Descent for Multi-task Learning.** NeurIPS 2021.  
   https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html

4. Navon A, Shamsian A, Achituve I, et al. **Multi-Task Learning as a Bargaining Game.** ICML 2022.  
   https://proceedings.mlr.press/v162/navon22a.html

5. Senushkin D, Patakin N, Kuznetsov A, Konushin A. **Independent Component Alignment for Multi-Task Learning.** CVPR 2023.  
   https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html

6. Liu B, Feng Y, Stone P, Liu Q. **FAMO: Fast Adaptive Multitask Optimization.** NeurIPS 2023.  
   https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract.html

7. Ban H, Ji K. **Fair Resource Allocation in Multi-Task Learning.** ICML 2024.  
   https://proceedings.mlr.press/v235/ban24a.html

8. Ban H, Subramani GR, Ji K. **SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation.** ICCV 2025.  
   https://openaccess.thecvf.com/content/ICCV2025/html/Ban_SAMO_A_Lightweight_Sharpness-Aware_Approach_for_Multi-Task_Optimization_with_Joint_ICCV_2025_paper.html

---

# 参考实现

- PCGrad official:  
  https://github.com/tianheyu927/PCGrad

- PCGrad PyTorch reimplementation:  
  https://github.com/WeiChengTseng/Pytorch-PCGrad

- CAGrad official:  
  https://github.com/Cranial-XIX/CAGrad

- Nash-MTL official:  
  https://github.com/AvivNavon/nash-mtl

- Aligned-MTL official:  
  https://github.com/taohong08/MTL

- FAMO official:  
  https://github.com/Cranial-XIX/FAMO

- FairGrad official:  
  https://github.com/OptMN-Lab/FairGrad

- SAMO official:  
  https://github.com/OptMN-Lab/SAMO

- LibMTL（可用于核对多种 MTL optimizer 的统一实现接口）：  
  https://github.com/median-research-group/LibMTL
