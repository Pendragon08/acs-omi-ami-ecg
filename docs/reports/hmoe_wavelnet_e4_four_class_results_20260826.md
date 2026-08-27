---
experiment_id: ACS-HMOE-WAVELNET-E4-MULTICLASS-20260826
date: 2026-08-26
status: completed
task: mutually exclusive four-class ACS classification
classes: OTHER, STEMI, NSTEMI, UA
model: E4 (Xception morphology + LeadSpatial GAT + WavelNet + task routers + KAN heads)
loss: CrossEntropyLoss (unweighted)
seed: 2026
fold_seed: 2026
device: NVIDIA RTX A6000 GPU1
decision_rule: argmax(softmax)
---

# E4 四类 Softmax 分类实验（修正版）

## 1. 修改原因

原 E4 使用三个独立 sigmoid 输出（STEMI/NSTEMI/UA），形式上属于三标签多输出；但当前 ACS 训练数据中没有重叠阳性标签，实际观测到的是四个互斥状态：

| 状态 | 含义 | 记录数 |
|---|---|---:|
| `000` | OTHER/阴性 | 9,070 |
| `100` | STEMI | 1,442 |
| `010` | NSTEMI | 1,235 |
| `001` | UA | 6,213 |

因此本次修改保留 E4 专家、路由、预处理和训练协议，仅将任务头改为四类 softmax，并使用未加权 `CrossEntropyLoss`。历史三 sigmoid/Youden 阈值结果保留在 [hmoe_wavelnet_e4_youden_threshold_results_20260826.md](hmoe_wavelnet_e4_youden_threshold_results_20260826.md)，没有覆盖。

## 2. 数据与固定实验协议

- 数据：17,960 条 ECG 记录、17,018 位患者；标签由三列 STEMI/NSTEMI/UA 映射为 `OTHER/STEMI/NSTEMI/UA`。
- 划分：沿用历史 `patient-level joint-label stratified 5-fold`，`fold_seed=2026`，不重新改变患者划分；每折约 14,3xx/3,5xx 条记录。
- 预处理：同一 P2 0.5–40 Hz ECG；每折仅用训练折拟合 robust scaler；增强仅用于训练折。
- 模型：E4（Xception morphology、LeadSpatial GAT、WavelNet、task routers、KAN heads），输出维度由 3 改为 4；参数量 382,446。
- 优化：Adam，学习率 `1e-4`，batch size `128`，最多 100 epochs，early stopping patience `15`，梯度裁剪 `1.0`，无 scheduler，路由平衡项 `0.01`。
- 选择：以验证折 macro AUROC 保存最佳 epoch；测试集仅在五个 checkpoint 固定后进行集成推理。
- 决策：`argmax(softmax)`，不再为三个标签分别寻找阈值；因此每条记录严格只有一个四类结果。为兼容旧外部平台，`OTHER` 编码为 STEMI/NSTEMI/UA 全 0。

## 3. GPU1 运行结果

本次五折运行已在 GPU1 完成（单次 seed `2026`；这是五个 patient-level validation folds，不是五个随机 seed）。下表是五折宏平均的 `mean ± SD`（按运行脚本 `np.std`，`ddof=0`）：

| 指标 | mean ± SD |
|---|---:|
| AUROC | 0.771175 ± 0.002110 |
| AUPRC | 0.519229 ± 0.006535 |
| Sensitivity | 0.491480 ± 0.009596 |
| Specificity | 0.808652 ± 0.003653 |
| PPV | 0.506951 ± 0.006097 |
| NPV | 0.821184 ± 0.004680 |
| F1 | 0.486419 ± 0.006659 |
| MCC | 0.312383 ± 0.005536 |
| Accuracy（四类宏平均） | 0.781338 ± 0.003771 |

### 3.1 OOF 汇总（17,960 条）

OOF 是五个验证折拼接后的 pooled 结果；它与上面的 fold mean 不完全相同。

| 类别 | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OTHER | 0.687752 | 0.675220 | 0.776516 | 0.474241 | 0.601092 | 0.675316 | 0.677635 | 0.263271 | 0.626893 | 7043/4216/4674/2027 |
| STEMI | 0.934678 | 0.658586 | 0.643551 | 0.962526 | 0.599871 | 0.968683 | 0.620943 | 0.587015 | 0.936915 | 928/15899/619/514 |
| NSTEMI | 0.800277 | 0.268295 | 0.252632 | 0.962392 | 0.331562 | 0.945766 | 0.286765 | 0.244197 | 0.913586 | 312/16096/629/923 |
| UA | 0.656010 | 0.460616 | 0.293417 | 0.835532 | 0.485486 | 0.690954 | 0.365770 | 0.150837 | 0.647996 | 1823/9815/1932/4390 |
| **四类宏平均** | **0.769679** | **0.515679** | **0.491529** | **0.808673** | **0.504503** | **0.820180** | **0.487778** | **0.311330** | **0.781347** | — |

四类 exact accuracy 为 `0.562695`，macro-F1 为 `0.487778`，weighted-F1 为 `0.538321`。主要瓶颈是 NSTEMI/UA 的召回率（分别为 0.252632 和 0.293417），所以该未加权 CE 版本目前只能证明修正后的四类流程可运行，不能宣称性能提升。

### 3.2 OOF 四类混淆矩阵

行是真实类别，列是预测类别，顺序均为 `OTHER, STEMI, NSTEMI, UA`：

```text
[[7043, 278, 230, 1519],
 [ 259, 928, 138,  117],
 [ 454, 173, 312,  296],
 [3961, 168, 261, 1823]]
```

### 3.3 五折宏指标逐折明细

| Fold | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.771140 | 0.525677 | 0.488120 | 0.802398 | 0.513626 | 0.826101 | 0.474782 | 0.312358 | 0.780356 |
| 1 | 0.774613 | 0.508640 | 0.491805 | 0.810935 | 0.506749 | 0.824496 | 0.486504 | 0.315240 | 0.785556 |
| 2 | 0.768128 | 0.520837 | 0.502263 | 0.813045 | 0.505349 | 0.823770 | 0.493888 | 0.321032 | 0.784882 |
| 3 | 0.770261 | 0.525694 | 0.475303 | 0.807220 | 0.512477 | 0.817921 | 0.485135 | 0.305167 | 0.780847 |
| 4 | 0.771733 | 0.515298 | 0.499908 | 0.809664 | 0.496552 | 0.813633 | 0.491787 | 0.308119 | 0.775049 |

## 4. 测试集推理与提交文件

测试集共 `1,995` 条记录。五折概率平均后取四类 argmax，预测类别计数为：`OTHER=1,375`、`STEMI=168`、`NSTEMI=85`、`UA=367`。多阳性数为 `0`；预测为 OTHER 的记录（兼容三列提交中的全 0）为 `1,375`。

本地同步的逐记录概率和提交文件：

- [test_predictions.csv](../.remote_project_scan/results/hmoe_kan_e4_multiclass/test_predictions.csv)
- [submission_stemi_nstemi_ua_e4_multiclass.csv](../.remote_project_scan/results/hmoe_kan_e4_multiclass/submission_stemi_nstemi_ua_e4_multiclass.csv)

服务器目录：

`/data/code/ouyanghao/oyh/ACS-OMI-AMI/results/hmoe_kan_e4_multiclass/`

本次只生成了兼容外部平台的提交文件，尚未把它与外部测试平台返回的指标混写；外部测试集真实标签没有用于训练、阈值或模型选择。

## 5. 修改结论

这次修改解决了任务定义上的流程错误：当前数据应按四类互斥状态建模，而不是把互斥标签当成可同时为真的三个 sigmoid 标签。旧三标签模型出现多阳性预测（历史测试输出中 698/1,995 条至少两个阳性），四类 softmax 修正后严格为单一类别。

但本次使用的是未加权 `CrossEntropyLoss`，NSTEMI/UA 的 sensitivity 和 F1 仍偏低。因此它是“语义正确、流程可复现”的基线修正版，不是最终临床性能方案。后续如要提高少数类召回，应在这个四类固定划分上单独比较 class-weighted CE、CB-Focal 或代价敏感 loss；不能把那些改变与本次任务形式修改混为同一实验。

## 6. 可追溯文件

- 配置：[e4_multiclass.yaml](../configs/e4_multiclass.yaml)
- 训练代码：[hmoe_kan_e4_multiclass.py](../.remote_project_scan/experiments/hmoe_kan_e4_multiclass.py)
- 汇总：[summary.json](../.remote_project_scan/results/hmoe_kan_e4_multiclass/summary.json)
- 划分：[split.json](../.remote_project_scan/results/hmoe_kan_e4_multiclass/split.json)
- OOF 概率：[oof_probabilities.npz](../.remote_project_scan/results/hmoe_kan_e4_multiclass/oof_probabilities.npz)
- OOF 逐记录结果：[oof_predictions.csv](../.remote_project_scan/results/hmoe_kan_e4_multiclass/oof_predictions.csv)
- 五折训练历史：`history_fold0.json`–`history_fold4.json`
- 五折 checkpoint：`e4_multiclass_fold0_seed2026.pt`–`e4_multiclass_fold4_seed2026.pt`
- 哈希和运行参数：[e4_multiclass_provenance_20260826.json](e4_multiclass_provenance_20260826.json)
