# ACS HMoE/KAN 可行性实验报告

状态：GPU1 E0→E4 实验进行中（2026-08-25）。最终结果将在全部五折完成后回填；未完成的实验不提前下结论。

## 方案修订

Frequency Expert 按 WavelNet 论文改为端到端 WaveletConv：12 导联共享 16 个 Symlet-4 小波滤波器（kernel=103，fs=100 Hz），随后使用轻量 Conv/MaxPool 编码器。不使用 MVMD/VMD 外部预处理或缓存。

论文：[WavelNet](https://doi.org/10.1016/j.cmpb.2023.107375)；官方实现：[NamhoKim-Kor/WavelNet](https://github.com/NamhoKim-Kor/WavelNet)。

## 数据与划分

- 训练记录：17,960；唯一患者：17,018；输入 `[N,1000,12]`，100 Hz。
- AMI 阳性 2,679（14.916%）；OMI 阳性 1,151（6.409%）。
- HMoE 实验：患者级、AMI/OMI 联合状态分层 5-fold，fold seed=2026；空的 `[AMI=0, OMI=1]` 状态不参与“最小样本数”报错检查。
- 输入链路保持 P3：P2（0.5–40 Hz）+ 每折训练集 robust scaler + 训练期增强；验证/外部测试不增强。

## 运行记录

| GPU | 实验 | 状态 |
|---|---|---|
| GPU0 | XceptionTime + P3 + BCE，STEMI/NSTEMI/UA，seed=2026 | 已完成；结果见 `results/stemi_nstemi_ua_final_gpu0/` |
| GPU1 | E0→E4，AMI/OMI，5-fold | 进行中；结果见 `results/hmoe_kan_github/` |

GPU0 验证集（threshold 由验证集 F1 选择）：

| Label | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 | MCC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STEMI | 0.9256 | 0.6696 | 0.6748 | 0.9681 | 0.6103 | 0.9758 | 0.6409 | 0.6138 | 0.30 |
| NSTEMI | 0.7839 | 0.2912 | 0.3583 | 0.9442 | 0.3162 | 0.9533 | 0.3359 | 0.2855 | 0.33 |
| UA | 0.6594 | 0.4688 | 0.8776 | 0.3302 | 0.4160 | 0.8322 | 0.5644 | 0.2271 | 0.16 |
| Macro | 0.7896 | 0.4765 | — | — | — | — | 0.5138 | — | — |

外部测试集本地没有标签，但平台支持盲评。使用固定验证阈值提交合并 CSV 后，平台返回：

| Label | Sensitivity | Specificity | PPV | NPV | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|
| STEMI | 0.7099 | 0.9624 | 0.6250 | 0.9740 | 0.9419 | 0.6647 |
| NSTEMI | 0.3235 | 0.9414 | 0.2876 | 0.9501 | 0.8992 | 0.3045 |
| UA | 0.8886 | 0.3382 | 0.4108 | 0.8538 | 0.5263 | 0.5619 |

提交文件：`results/stemi_nstemi_ua_final_gpu0/submission_stemi_nstemi_ua_XceptionTime.csv`；平台返回页面已保存为 `official_platform_stemi_nstemi_ua_XceptionTime.html`。平台反馈属于 development/official blind evaluation，不能当作本地可复算的有标签测试集统计。

## 结果

待 GPU1 全部 E0–E4 完成后填写每折结果、mean±std、Router 权重、WavelNet scale 统计和最终可行性判断。
