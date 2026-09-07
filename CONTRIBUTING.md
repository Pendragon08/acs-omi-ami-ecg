# 协作与安全发布说明

_ACS–OMI 研究仓库 · 更新于 2026-09-07_

---

## 👥 协作方式

已有写权限的成员从 `main` 创建独立分支，按代码、配置、实验凭据和中文报告一起提交 Pull Request。没有写权限时使用 fork / Pull Request。本文不修改 GitHub 成员权限。

```bash
git fetch origin
git switch -c docs/describe-change origin/main
git status --short
git diff --check
# 审阅后仅暂存本次明确修改的文件，禁止盲目添加全部目录。
git add <reviewed-file>
git diff --cached --stat
git diff --cached --check
```

不要 force push、重写公共历史或为了目录整洁移动训练中使用的路径。旧 checkpoint 依赖的模型接口必须兼容。

## 🔐 提交边界

可以提交：源代码、审阅后的配置、聚合指标、脱敏 epoch 曲线、可视化、测试记录和中文报告。

禁止提交：凭据、私钥、原始 ECG/WFDB、患者级标签/划分表、逐记录预测、外部平台提交文件、checkpoint 和下载权重。模型/数据的再分发还须遵守原许可。[DATASET.md](DATASET.md)记录数据使用边界。

`.gitignore` 不会自动移除已跟踪文件，也不保证自动发现改名后的患者表。发布时检查暂存区文件内容与 CSV/JSON 字段，不只检查扩展名。历史总调度日志和聚合 RL JSONL 的保留不构成新增原始日志的通行证；新日志先导出为安全的聚合产物再审阅。

## 🔍 结果审阅要求

- 明确区分 inner-validation、outer fold、pooled OOF、独立外部测试。
- 每个指标关联配置、seed、fold、checkpoint 选择规则、阈值来源及审计凭据。
- 在研候选注明“未完成”，不按个别中间指标宣布最佳。
- 修正报告时保留原指标/receipt，用附带 hash 的 sidecar 记录修订。
- `legacy_reporting_v1/` 只用于审计，不能再次汇总为正确训练曲线。
- 不将五折当作五个随机种子，不将历史整卡显存与 PyTorch allocated 峰值直接作统一成本比较。

## 🧪 测试与执行边界

当前相关回归测试需在已配置 ECG-LFM / fairseq 的环境中执行；本仓库不携带数据与模型权重。环境路径按本机配置，不应复制他人的凭据。

```bash
PYTHONPATH=.:third_party/ECG-LFM python -m unittest discover -s tests -p 'test_joint6_*.py'
PYTHONPATH=.:third_party/ECG-LFM python -m unittest discover -s tests -p 'test_ecg_lfm_joint6.py'
PYTHONPATH=.:third_party/ECG-LFM python -m unittest discover -s tests -p 'test_ecg_lfm_corrected_evaluation.py'
```

长任务必须使用 `screen`、写独立日志、以 `CUDA_VISIBLE_DEVICES` 绑定一张 GPU。更新文档或发布 Git 不等于获准重跑实验、重新划分患者或停止现有进程。

目前报告兼容审计固定了 evaluator 及其依赖的精确 hash。即使只是格式化这些文件，也可能使历史结果拒绝通过审计；修改后必须重新核验，不得放宽校验或伪造 receipt。
