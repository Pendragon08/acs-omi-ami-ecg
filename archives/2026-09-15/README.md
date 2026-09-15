# 服务器实验结果归档（2026-09-15）

本分支是现有项目的增量归档，不替换 main。状态：**完整模型归档、远端校验、真实恢复测试和服务器清理均已完成**。

## 已保存的内容

- `aggregate_metrics.json`：从305份实验报告提取的63,081个有限数值汇总指标，保留相对来源路径和源文件SHA-256。只导出标量，不导出患者行、数组、自由文本和标识符；不是完整原报告的替代。
- `model_inventory.json`：服务器 results 下413份训练模型的待归档清单，约126.18GiB。
- `restore_verification.json`：从GitHub实际下载分块并完整重建一份模型，字节数与SHA-256均相同。
- 本目录脚本：内容去重、上传、校验与恢复。不会反序列化PyTorch pickle。

大型模型保存在本私有仓库的 [Release](https://github.com/Pendragon08/acs-omi-ami-ecg/releases/tag/server-model-archive-20260915)。GitHub普通文件历史不存放模型；每个ZIP分块远小于2GiB。备份按原始ZIP存储边界分块去重，不改变模型张量精度、不删除优化器状态，恢复目标是整个原文件逐字节相同。

## 保留在服务器的内容

原始ECG数据、患者级预测/划分表、原始日志、下载的预训练checkpoint、代码和配置暂不删除。只会在完整备份、远端校验和恢复测试通过后删除**清单中已备份的训练模型文件**，不会递归删除整个项目或results目录。

## 完成后的恢复方式

需要对此私有仓库有读取权限，以及足够的恢复磁盘空间。只有Release存在最终 `model-manifest.json` 且归档完成记录确认后，才按以下方式恢复。

```sh
gh release download server-model-archive-20260915 --repo Pendragon08/acs-omi-ami-ecg --pattern 'model-pack-*.zip' --pattern model-manifest.json --dir archive-packs
python restore_models.py --manifest archive-packs/model-manifest.json --packs archive-packs --output restored-project
```

可用 `--only results/某实验/checkpoint_best.pt` 恢复单个文件；先根据manifest的chunk→pack索引下载所需分块即可。脚本拒绝覆盖既有文件，校验分块与原文件SHA-256，不加载模型执行代码。原文件路径相对于项目根目录，恢复时会自动建立results子目录。

## 删除门槛

1. 每个分块经本地解压及SHA-256验证，GitHub API返回的digest与本地一致。
2. 最终manifest已经上传且校验通过；本分支记录完成清单和恢复说明。
3. 至少一次真实GitHub下载与完整模型重建测试通过。
4. 删除前再次核对源文件路径、大小、修改时间和完整SHA-256；发生变化即保留。
5. 保存逐文件清理凭据及实际磁盘可用空间变化。

上传失败、清单不全或校验不一致时，保留源文件。实际进度以Release与后续归档完成记录为准，不能仅凭此说明认定备份完成。


## 清理结果

已按校验清单删除 413 份训练模型；实际可用空间增加约 126.18 GiB。原始数据、患者级表格、日志、代码与下载checkpoint保留。详见 `archive_complete.json` 与 `cleanup_receipt.json`。
