# ECG-LFM 四标签微调实验记录（LEGACY—任务定义错误）

> **INVALID FOR CURRENT TASK (2026-08-29)**: the historical four-output BCE/sigmoid task treated ACS subtypes as independent. Do not use its metrics as final results; see the corrected joint ECG-LFM report.

更新时间：2026-08-27（Asia/Shanghai）  
状态：**实现、环境、原始数据、队列/切分、全量预处理、官方权重、真实前向、64-record 过拟合门禁及大 batch 单步门禁均通过；L0 Fold 0 正在 GPU1、L1a Fold 0 正在 GPU0 并行训练。**

> 这是 ECG-LFM 初始流水线阶段性记录；后续完整五折、LoRA/DoRA 和外部结果已在 [`ecg_lfm_local_best_lora_dora_optimization_20260828.md`](ecg_lfm_local_best_lora_dora_optimization_20260828.md) 中更新。

本报告是新 ECG-LFM 分支的唯一主记录。历史 E4 方案、模型、检查点和实验结果均未修改，继续保存在：

- `reports/hmoe_wavelnet_e4_stemi_nstemi_ua_omi_multilabel_20260827.md`

学习率、有效 batch、训练曲线与 fold-specific Weighted BCE 的后续独立消融记录在 `reports/ecg_lfm_hyperparameter_loss_ablation_20260827.md`；调参分支不会覆盖本报告中的原始 unweighted BCE baseline。

旧 E4 关键文件在启动新分支后再次取证：结果目录共 21 个文件；`e4_4label_summary.json`、`oof_predictions.csv`、`split.json` 的 SHA-256 分别为 `b72ca93fdaa530fb73ef11ab7ec6bb79d0ee2ed8317518bd95e6295c41714e60`、`f99fecbad6acb517e224dccb2bf437f427951d7efbf0f28c0146ded402d47f9a`、`63bd69e7d19811ef25074376efacb240500135fd6db824831ed2a99c110fc510`。新分支只读取这些文件用于 exact split 与配对比较，不向旧目录写入。
- `results/hmoe_kan_stemi_nstemi_ua_omi_e4/`

## 1. 任务定义

- 四个独立 sigmoid 二分类输出：`STEMI,NSTEMI,UA,OMI`。
- 保留明确的 subtype `[0,0,0]` non-ACS 负样本。
- 内部训练队列：17,960 ECG，17,018 patients。
- 隔离外部队列：1,995 ECG。
- 全队列：19,955 ECG。
- 直接读取历史 E4 `split.json` 中五组 `folds_indices`；禁止重新随机划分。
- 外部队列不参与训练、early stopping、checkpoint 选择、阈值选择或方案选择。

## 2. 新模型

新模型不是 HMoE/KAN 的改写，而是官方 ECG-LFM 单独迁移实验：

```text
raw 500 Hz, 10 s, 12-lead WFDB
  -> 1 Hz high-pass
  -> 50/60 Hz notch
  -> 30 Hz low-pass (2nd-order Butterworth)
  -> per-record/per-lead z-score
  -> two non-overlapping 5 s segments
  -> official ECG-LFM encoder
  -> official mean pooling
  -> official 1024 -> 4 linear classification head
  -> average the two segment logits at record level
  -> sigmoid probabilities
```

第一轮损失固定为未加权 `BCEWithLogitsLoss`。不加入 GAT、WavelNet、MoE、KAN、class weight、sampler 或数据增强，确保只验证 foundation representation。

## 3. 适配实验

| ID | 编码器策略 | 训练头 | 用途 |
|---|---|---|---|
| L0 | encoder frozen (`freeze_finetune_updates=999999999`) | 4-output linear head | linear probe gate |
| L1a | official-style fine-tune；conv feature extractor 保持官方 `feature_grad_mult=0` | 4-output linear head | 主方案 |
| L1b | 与 L1a 相同 | 4-output linear head | fixed scheduler 对照 |
| L2 | 尚未启动 | — | 仅在 L1 稳定后考虑的 unfreeze 消融 |

共同训练参数：Adam `betas=(0.9,0.98)`、`eps=1e-8`、LR `1e-6`、最多 100 epochs、validation-loss checkpoint、patience 10、effective batch 256、seed 2026。为充分使用 A6000 显存且保持每 epoch 的 optimizer updates 完全不变，L0 使用 physical batch 64 + gradient accumulation 4；L1a/L1b 使用 physical batch 32 + gradient accumulation 8。L1a 使用官方 step scheduler 实现每 112 updates 乘 `0.99` 的逐 epoch 指数衰减近似；L1b 保留官方 fixed scheduler。

## 4. 来源与完整性

| 资源 | 来源 | 固定标识 | 当前状态 |
|---|---|---|---|
| ECG-LFM code | https://github.com/biomed-AI/ECG-LFM | commit `647369a012dd8d5d362e99e2bf238c91f3e1d03b` | 已在服务器校验 |
| ECG-LFM checkpoint | https://zenodo.org/records/20388950 | `pretrain.pt`, MD5 `afa8dfc680590948f82aabbdb123bf43`, 7,522,279,891 bytes | **已完成并通过整文件大小/MD5 校验** |
| ACS raw WFDB | https://doi.org/10.6084/m9.figshare.29925314 | `ECG_row_data.zip`, MD5 `acea6ca86a2d0b937ecfe7d1df6d30a2` | 已下载、MD5 校验并解压；19,955 对 WFDB 文件 |
| historical split | `results/hmoe_kan_stemi_nstemi_ua_omi_e4/split.json` | patient-level iterative multilabel 5-fold, seed 2026 | exact join 与五折逐条复用门禁已通过 |

严禁以现有 100 Hz P2 上采样替代原始 500 Hz 数据。本次已使用官方原始 500 Hz 数据完成校验：归档大小 1,277,577,332 bytes，MD5 与官方一致；解压后 `.hea/.dat` 各 19,955 个。全量文本头审计结果为 `bad_header=0`、`bad_lead_order=0`、`pair_id_mismatch=0`，首条头信息为 12 导联、500 Hz、5,000 samples。

全量 WFDB API 物理信号解码进一步确认 19,955/19,955 均可读取，但发现并记录了官方数据自身的两个小异常：3 条记录各含 1 个 NaN（合计 3 samples），另有 `03228`、`14262` 两条 `.dat` 实际为 3,500 samples，而 `.hea` 写成 5,000。无 Inf、无缺导联、其余 19,953 条均为 5,000 samples。处理规则直接复用数据作者官方 `validation-code` commit `d80225998e2c80cdb49069efaefea1224bb7a98e`：NaN 在滤波前置零，短记录在滤波前零填充到 5,000；原始归档不改写、样本不删除，逐条异常字段保存在 `artifacts/ecg_lfm/raw_500hz_manifest.csv`。

完整预处理最终结果：19,955 records 全部成功、39,910 个 5 秒 segment 齐全、0 errors，耗时 57.29 s。另识别出 52 条记录共 55 个恒定导联；严格采用官方 ECG-LFM `scripts/preprocess/ecg/preprocess.py` 默认 `constant_lead_strategy=zero`，将零方差导联视作 masked/absent lead 并保持为零。逐条 record/lead 名称保存在 `manifests/ecg_lfm/preprocess_integrity.json`，没有删除或改变切分。

队列/切分独立门禁结果：内部 17,960 ECG / 17,018 patients，外部 1,995 ECG；四标签阳性数依次为 STEMI 1,442、NSTEMI 1,235、UA 6,213、OMI 1,151；明确保留 subtype `[0,0,0]` 9,070 条。五个验证折大小为 3,606 / 3,582 / 3,596 / 3,603 / 3,573，全部满足 train/validation index overlap=0、patient overlap=0，且 OOF 精确覆盖 17,960 条。

## 5. 已实现文件

- `configs/ecg_lfm/env-spec.json`
- `configs/ecg_lfm/linear_probe.yaml`
- `configs/ecg_lfm/finetune_paper.yaml`
- `configs/ecg_lfm/finetune_fixed.yaml`
- `data/ecg_lfm_manifest.py`
- `data/ecg_lfm_preprocess.py`
- `models/ecg_lfm_adapter.py`
- `experiments/ecg_lfm/run_fold.py`
- `experiments/ecg_lfm/infer.py`
- `experiments/ecg_lfm/smoke_forward.py`
- `experiments/ecg_lfm/contract_tests.py`
- `experiments/ecg_lfm/run_overfit_gate.py`
- `experiments/ecg_lfm/verify_overfit_gate.py`
- `experiments/ecg_lfm/verify_fold0_gate.py`
- `data/ecg_lfm_gate_manifest.py`
- `scripts/download_ecg_lfm.sh`
- `scripts/download_ecg_lfm_checkpoint_parallel.sh`
- `scripts/run_ecg_lfm_5fold.sh`
- `scripts/run_ecg_lfm_post_checkpoint_gates.sh`
- `scripts/run_ecg_lfm_after_gates.sh`
- `scripts/resume_ecg_lfm_l1a_after_fold0.sh`
- `scripts/evaluate_ecg_lfm.py`
- `scripts/ensemble_ecg_lfm.py`
- `scripts/compare_ecg_lfm_to_e4.py`
- `scripts/submit_hmoe_e4_4label.py`
- `experiments/ecg_lfm/README.md`
- `.aris/compute/nlp03.md`
- `artifacts/ecg_lfm/conda_explicit_linux-64.txt`
- `artifacts/ecg_lfm/pip_freeze.txt`
- `artifacts/ecg_lfm/gpu_inventory.csv`

所有新增路径与旧 E4 分离。服务器端 Python `py_compile` 已通过。官方源码远端 commit 已校验；本地克隆保留完整 `.git` 历史。

源码工作树另做了行尾敏感性审计：Windows→Linux 同步产生了 CRLF/LF-only 的 Git worktree 差异，因此原始 full diff SHA-256 为 `3abb91c6d7f0ffdce3e05a79bcfa8badf3eb55e4f998717c02d4e5c33365a860`；使用 `git diff --ignore-space-at-eol` 后，唯一有非零 numstat 的文件是 `fairseq_signals/utils/checkpoint_utils.py`（13 insertions / 10 deletions），即下述已记录的 strict-best 保存修复，semantic diff SHA-256=`52a2bad8b670992bbc39cc6f5f71310e438252ea21ef726dece237c7a0b91948`。模型、criterion、optimizer、scheduler、预处理与评价源码相对固定 commit 均无语义 diff；机器可读取证为 `artifacts/ecg_lfm/official_source_integrity.json`。

## 6. 强制执行 gates

| Gate | 验收条件 | 状态 |
|---|---|---|
| G1 | raw archive MD5 正确 | **Passed** |
| G2 | 19,955 对 `.dat/.hea`；WFDB readable；500 Hz；12 leads；异常有记录且按官方规则处理 | **Passed（全量 19,955 WFDB API 解码；3 个 NaN + 2 条 3,500 点短记录，见审计）** |
| G3 | raw/CSV/meta/split 按 record ID 完全 join；train=17,960、external=1,995 | **Passed** |
| G4 | 五折索引逐条复用且 patient overlap=0；OOF 精确覆盖 17,960 | **Passed** |
| G4b | paper-native 预处理产生 19,955 × 2 segments；无缺失/NaN/Inf 输出 | **Passed（39,910 files，0 errors）** |
| G5 | checkpoint MD5 与官方一致，commit 已记录 | **Passed（7,522,279,891 bytes；`afa8dfc680590948f82aabbdb123bf43`）** |
| G6 | single-record forward 输出 `[1,4]` | **Passed（GPU0 / RTX A6000；finite）** |
| G7 | 两个 segment logits 聚合为一条 record probability | **Passed（模型 witness: `[1,2,4] -> [1,4]`）** |
| G8 | small-subset overfit | **Passed（valid loss 0.475 → 0.271，下降 42.95%）** |
| G8b | 大 batch 单个 optimizer update，无 OOM/NaN 且有效 batch 保持 256 | **Passed（L0 64×4，训练后剩余 33.1 GiB；L1a 32×8，训练后剩余 28.2 GiB）** |
| G9 | Fold 0 训练稳定 | **Running**：L1a epoch 5 segment-level valid AUROC/AUPRC/F1@0.5=`0.8697/0.4886/0.3779`，loss=`0.412`；L0 epoch 14=`0.5539/0.1429/0.1208`，两者均已越过 segment-level 最低 AUROC 健康下限，继续按预注册 validation-loss/early-stopping 选最佳 checkpoint，结束后再做正式 record-level gate |
| G10 | L0/L1a full 5-fold + external ensemble | Pending |

## 7. 服务器运行记录

服务器：受控 GPU 执行环境（内部地址不纳入版本库）  
代码目录：`/data/code/ouyanghao/oyh/ACS-OMI-AMI`  
GPU：2 × RTX A6000 48 GiB；启动时 GPU0/GPU1 均空闲。

运行记录：

- `ecg_lfm_env_build`：首次构建已装好 Python 3.9 与 PyTorch 2.5.1+cu121，但新版 pip 拒绝 OmegaConf 2.0.6 的历史元数据写法；环境规范补充固定 `pip==24.0` 后已成功完成。Tier-1 imports、官方 `fairseq-hydra-train --help`（144 行）和 GPU0 seeded CUDA witness 均通过；关键版本为 PyTorch 2.5.1+cu121、WFDB 4.1.2、SciPy 1.13.1、ECG-LFM `1.0.0a0+647369a`、OmegaConf 2.0.6。日志 `logs/ecg_lfm/env_build.log`。
- `ecg_lfm_ckpt_dl`：32 路官方 Zenodo range 下载、合并、MD5 已完成；官方权重大小与 MD5 均一致。日志 `logs/ecg_lfm/checkpoint_download.log`。
- `ecg_lfm_gates`：等待最终 checkpoint 出现并通过 MD5 后，在 GPU0 自动顺序执行 single-record forward、64-record 过拟合诊断及其验收；日志 `logs/ecg_lfm/post_checkpoint_gates.log`。
- `ecg_lfm_pipeline`：仅在过拟合门禁与大 batch 单步门禁通过后启动；L0 Fold 0 在 GPU1、L1a Fold 0 在 GPU0 并行执行，随后分别运行覆盖率/概率非退化/最低 macro-AUROC 门禁。L1a Fold 0 完成并通过 record-level gate 后，`ecg_lfm_l1a_resume` 会立即复用释放的 GPU0 推进 folds 1–4；主流水线和恢复流水线通过 `/tmp/ecg_lfm_${USER}_gpu*.lock` 与 adaptation/fold 级 `flock` 串行化冲突作业，重复入口只会检查并跳过已完成折，不会重复训练。L0 是可选 representation probe，若其 Fold 0 未达门禁会如实停止 L0 full fivefold，但不会阻断主方案 L1a。正式日志为 `logs/ecg_lfm/after_gates_large_batch.log`、`logs/ecg_lfm/l1a_fold0_large_batch.log` 与 `logs/ecg_lfm/l1a_resume.log`。
- `ecg_lfm_preprocess`：已完成；8 workers 生成 19,955 × 2 个 5 秒 MAT segment；日志 `logs/ecg_lfm/preprocess_full.log`。

环境 spec canonical hash：`433ea417`。原单连接 Zenodo 下载仅约 0.1–0.2 MB/s，临时 partial 文件被保留为 `pretrain.pt.single_connection_partial_20260827`，之后改为 32 路 range 下载；这一变更只影响下载速度，不改变字节内容，最终仍必须通过官方 MD5。Figshare 命令行曾遭遇 AWS WAF 202/403，最终通过正常浏览器会话取得官方签名 S3 下载并完成官方 MD5，未使用任何非官方镜像。

环境还保存了可直接复现的 Conda explicit lock、pip freeze 与 GPU/driver 清单；前两者 SHA-256 分别为 `8e1e98542f0f617e1df624cb0fc89554bee0f269de7a2f03f2917bc23179d295` 和 `e63c6ea4eb7ca198ec9a2d2b923e430667301989be2757cfcdac1e2f5c2df11a`。

官方 checkpoint 的 seeded forward witness：single segment `[1,4]`，two-segment tensor `[1,2,4]`，record aggregation `[1,4]`，finite=`true`，GPU=`NVIDIA RTX A6000`；固定 seed 2026 后连续两次运行的四标签概率均为 `[0.150061, 0.651709, 0.292322, 0.610563]`。这些数值只用于接口/数值稳定性与可复现门禁，不作为科学结果。

过拟合门禁使用相同 64 records / 128 segments 作为 train/valid 的纯软件诊断，10 epochs、160 updates、head-only LR `1e-3`，不保存模型且不进入科学指标。valid loss 从 0.475 降至 0.271（相对下降 42.95%），train loss 从 0.551 降至最低 0.321，验收通过。官方日志中的 `valid_nsignals≈8` 是聚合字段显示的 batch 大小；manifest 为 129 行（1 root + 128 samples），每 epoch 16 updates。

正式训练前分别完成一个 optimizer update 的显存门禁：L0 `physical 64 × accumulation 4`，L1a `physical 32 × accumulation 8`，两者有效 batch 均为 256，均无 OOM/NaN；训练步完成后 GPU 空闲显存分别约为 33.1 GiB 与 28.2 GiB。早期以 `physical 8 × accumulation 32` 启动的未完成 Fold 0 被完整移动到 `results/ecg_lfm_4label/partial_runs/physical8_20260827_1338/`，只作工程取证，不纳入正式科学结果；它未改写旧 E4 或本次正式结果目录。

正式 Fold 0 于服务器 UTC `2026-08-27 13:37` 并行启动。当前训练器验证轨迹显示：L1a 从 epoch 1 到 epoch 5 的 segment-level valid loss `0.452→0.412`、AUROC `0.8325→0.8697`、AUPRC `0.4135→0.4886`，数值稳定并已达到 Fold 0 最低 AUROC 健康下限；训练器固定 0.5 阈值下的 F1 在 epoch 4/5 为 `0.3882/0.3779`，该波动不参与 checkpoint 或最终阈值选择。L0 到 epoch 14 为 `loss/AUROC/AUPRC/F1@0.5=0.769/0.5539/0.1429/0.1208`，首次越过 0.55 健康下限，但排序指标仍明显弱于 L1a，只作为线性探针消融。以上只作为训练健康度记录，不能替代最佳 checkpoint 的两段 logit 聚合后 record-level 指标，亦不写入最终五折结果表。

门禁首次运行还暴露了官方 `checkpoint_utils.save_checkpoint` 的保存错误：函数先更新 `best` 再用非严格比较判断，导致每个 epoch 都被视作 best，并复制 metric-best / best / last 三份。最小运行时 patch `patches/ecg_lfm/checkpoint_strict_improvement.patch` 将条件改为相对 previous-best 的严格改进；独立契约以 loss `0.50 → 0.60 → 0.40` 验证只保存两次、最终 best=0.40。正式配置同时使用 `no_last_checkpoints=true`、`keep_best_checkpoints=-1`、`no_save_optimizer_state=true`，只保存推理所需的 model-only strict-best；不改变模型、loss、optimizer 更新、scheduler、seed、数据或评价。旧诊断产生的 4 个冗余 checkpoint 共 29,900,434,782 bytes 已删除，日志与 provenance 以 `*_pre_storage_fix` / `*_interrupted_storage_bug` 名称保留；删除文件均为可重新生成且不参与科学结果的门禁模型。

## 8. 评价方案

每标签输出：AUROC、AUPRC、Sensitivity、Specificity、PPV、NPV、F1、MCC、Accuracy、TP/TN/FP/FN。另输出 macro AUROC/AUPRC/F1 等、五折逐折指标、pooled OOF 指标，以及专门的 NSTEMI-vs-UA：AUROC、AUPRC、macro F1、balanced accuracy、两类 sensitivity、confusion matrix。

分类阈值：仅在完整 OOF probabilities 上对每个标签独立网格搜索 `0.01–0.99` 的 Youden J；同时保留 threshold 0.5 指标。单条 ECG 内先平均两个 segment logits 再 sigmoid；外部集则按实施文档对五折 record-level probabilities 做算术均值，再固定使用 OOF 阈值。外部标签不在本地使用；预测文件完成后再提交官方平台。

与 E4 的差异检验在完全相同的 17,960 条 OOF 记录上进行，按 patient ID 做 2,000 次 paired cluster bootstrap；四个单标签及 macro 均输出 ΔAUROC、ΔAUPRC、ΔSensitivity、ΔSpecificity、ΔF1 与 95% CI。每个 macro bootstrap replicate 对四标签共用同一批重采样患者，避免把 ECG 当作彼此独立样本。

## 9. 结果表（尚未训练，不填造结果）

| Model | Macro AUROC | Macro AUPRC | Macro F1 | STEMI AUROC | NSTEMI AUROC | UA AUROC | OMI AUROC | NSTEMI-vs-UA AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical E4 | 0.8208 ± 0.0049 | 0.4503 ± 0.0076 | 0.4147 ± 0.0256 | 见历史报告 | 见历史报告 | 见历史报告 | 见历史报告 | 见历史报告 |
| ECG-LFM L0 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| ECG-LFM L1a | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

备注：E4 行是历史五折 Youden mean ± SD，用于定位旧结果，不表示与尚未运行的 ECG-LFM 已完成统计比较。

## 10. 下一次记录点

1. 环境 tier-1 imports 与 seeded GPU witness。
2. checkpoint 合并后 size/MD5。
3. single-record 模型 forward、subset overfit、Fold 0 gate。
4. L0/L1a 五折、OOF 阈值、外部平台结果及 E4 同表比较。
