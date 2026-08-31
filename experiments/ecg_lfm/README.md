# ECG-LFM 联合任务实现

本目录只保留当前可执行的 ECG-LFM 联合流程。临床任务由两部分组成：

- ACS 阳性患者的亚型为 `STEMI / NSTEMI / UA` 单标签三分类；
- OMI/NOMI 为独立二分类。

由于训练全队列明确保留 `[0,0,0]` 的 non-ACS 记录，工程实现使用
`OTHER / STEMI / NSTEMI / UA` 四状态 softmax 头，再加一个 OMI sigmoid
logit。`OTHER` 是全队列状态，不是第四种 ACS 亚型。报告同时给出全队列状态
结果与限定 ACS 阳性样本后的三分类结果。

当前入口：

- `run_joint_fold.py`：运行一个患者级交叉验证折；
- `evaluate_joint.py`：汇总五折 OOF、选择 OMI 阈值、生成外部提交和图表；
- `adapter_contract_tests.py`：验证 LoRA/DoRA 注入与参数统计；
- `smoke_forward.py`：检查官方 ECG-LFM 前向接口；
- `write_adapter_artifacts.py`：导出脱敏参数统计。

配置仅使用 `configs/ecg_lfm/corrected_joint_{baseline,lora,dora}.yaml`。旧四个
独立 sigmoid 的启动器、阈值代码和调参脚本已从当前分支删除，可通过 Git 历史
恢复，但不得用于当前性能结论。

原始 ECG、患者级表格、checkpoint、逐例预测和提交 CSV 不进入 GitHub。
