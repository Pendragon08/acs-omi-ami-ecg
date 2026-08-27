"""Insert the completed loss/feature ablation summaries into FINAL_REPORT.md."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/loss_feature_ablation"
START = "<!-- LOSS_FEATURE_ABLATION_START -->"
END = "<!-- LOSS_FEATURE_ABLATION_END -->"


def main():
    report_path = ROOT / "FINAL_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    loss = (RESULTS / "loss_ablation_summary.md").read_text(encoding="utf-8")
    feature = (RESULTS / "feature_ablation_summary.md").read_text(encoding="utf-8")
    bootstrap = json.loads((RESULTS / "baseline_vs_cb_focal_paired_bootstrap.json").read_text(encoding="utf-8"))
    selection = json.loads((RESULTS / "stage1_selection.json").read_text(encoding="utf-8"))
    block = "\n".join(
        [
            START,
            "",
            "## 9. OMI CB-Focal 与输入特征消融（2026-08-18）",
            "",
            "本轮固定第 8 节已复现的历史患者级划分。Loss 实验只改变 criterion；特征实验使用同一个 XceptionTimeFusion 并屏蔽输入分支。所有选择均只使用 validation，未读取或提交外部测试反馈。",
            "",
            loss,
            "",
            feature,
            "",
            "### 统计与证据边界",
            "",
            f"- 最佳 CB-Focal validation 参数：beta={selection['best_cb_focal']['beta']}，gamma={selection['best_cb_focal']['gamma']}。",
            f"- paired patient-cluster bootstrap 共 {bootstrap['auprc']['n_bootstrap']} 次；详细结果见 `results/loss_feature_ablation/baseline_vs_cb_focal_paired_bootstrap.json`。",
            "- 外部测试集标签隐藏且历史 leaderboard 已被适应性访问；本节结果属于内部 validation 消融证据。只有方案完全冻结后的一次外部评估才能作为外部支持，仍不能替代新队列确认。",
            "- 历史 P3 scaler 包含内部 validation 的已知泄漏在所有组中保持一致，因此不解释组间差异，但会使绝对性能偏乐观。",
            "",
            END,
        ]
    )
    if START in report and END in report:
        before = report.split(START, 1)[0].rstrip()
        after = report.split(END, 1)[1].lstrip()
        report = before + "\n\n" + block + "\n\n" + after
    else:
        appendix = "## 附录：复现命令"
        if appendix in report:
            report = report.replace(appendix, block + "\n\n---\n\n" + appendix, 1)
        else:
            report = report.rstrip() + "\n\n---\n\n" + block + "\n"
    report_path.write_text(report, encoding="utf-8")
    print(f"updated {report_path}")


if __name__ == "__main__":
    main()
