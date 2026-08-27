# ACS 实验报告索引

| 报告 | 内容 | 状态 |
|---|---|---|
| [hmoe_wavelnet_e4_stemi_nstemi_ua_omi_multilabel_20260827.md](hmoe_wavelnet_e4_stemi_nstemi_ua_omi_multilabel_20260827.md) | 当前最终口径：全队列 STEMI/NSTEMI/UA/OMI 四标签独立二分类；GPU1 患者级五折、OOF 阈值与外部平台结果 | 已完成 |
| [e4_stemi_nstemi_ua_omi_multilabel_provenance_20260827.json](e4_stemi_nstemi_ua_omi_multilabel_provenance_20260827.json) | 当前四标签实验的命令、数据/代码/产物 SHA-256 与外部平台提交溯源 | 已完成 |
| [stemi_nstemi_ua_multilabel_results_20260826.md](stemi_nstemi_ua_multilabel_results_20260826.md) | GPU0 XceptionTime + P3 的 STEMI/NSTEMI/UA 三标签验证集与外部平台结果；GPU1 AMI/OMI E0–E4 汇总 | 已完成 |
| [hmoe_wavelnet_stemi_nstemi_ua_e4_results_20260826.md](hmoe_wavelnet_stemi_nstemi_ua_e4_results_20260826.md) | GPU1 E4（WavelNet + HMoE + KAN）三标签五折验证、OOF 阈值及外部平台盲评 | 已完成 |
| [acs_hmoe_kan_feasibility.md](acs_hmoe_kan_feasibility.md) | HMoE/KAN/WavelNet 方案说明及早期 AMI/OMI 可行性记录 | 历史记录 |

## 当前关键外部结果

GPU1 E4 五折集成提交到外部平台后的逐标签 F1：STEMI `0.6977`、NSTEMI `0.3146`、UA `0.5636`；逐标签宏平均 F1 `0.5253`。

提交文件和平台返回页面保存在服务器：

`/data/code/ouyanghao/oyh/ACS-OMI-AMI/results/hmoe_kan_stemi_nstemi_ua_e4/`

## 2026-08-26 revised sparse HMoE-KAN feasibility

[acs_sparse_hmoe_kan_feasibility_gpu1_20260826.md](acs_sparse_hmoe_kan_feasibility_gpu1_20260826.md) records the completed GPU1 five-fold run for the revised four-expert model (Xception + InceptionTime + Lead-GAT + Morlet CWT-CNN), three class-specific Noisy Top-2 routers, KAN heads, OOF thresholds, NSTEMI-vs-UA pairwise metrics, calibration, and router diagnostics. The result is a reproducible feasibility run; it does not yet demonstrate an improvement over the previous three-expert E4.

Server artifacts:

`/data/code/ouyanghao/oyh/ACS-OMI-AMI/results/acs_sparse_hmoe_kan_fullcohort_gpu1_20260826/`
| [hmoe_wavelnet_e4_youden_threshold_results_20260826.md](hmoe_wavelnet_e4_youden_threshold_results_20260826.md) | E4 Youden J / balanced-accuracy threshold OOF results | completed |
| [e4_youden_provenance_20260826.json](e4_youden_provenance_20260826.json) | E4 最终 Youden 阈值实验的可追溯 provenance、代码/数据路径和 SHA-256 校验值 | completed |
| [hmoe_wavelnet_e4_four_class_results_20260826.md](hmoe_wavelnet_e4_four_class_results_20260826.md) | E4 四类 Softmax（Other/STEMI/NSTEMI/UA）修正方案及 GPU1 五折运行记录 | completed |
| [e4_multiclass_provenance_20260826.json](e4_multiclass_provenance_20260826.json) | E4 四类 Softmax 运行参数、数据映射、产物路径和 SHA-256 溯源 | completed |
