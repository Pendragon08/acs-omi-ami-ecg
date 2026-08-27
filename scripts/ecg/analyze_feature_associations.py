"""Train/validation-only association analysis for age and sex versus OMI/AMI."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/loss_feature_ablation"


def aligned_frame():
    meta = np.load(ROOT / "data/arrays/meta_train.npz", allow_pickle=True)
    split = json.loads((ROOT / "data/splits/patient_split.json").read_text(encoding="utf-8"))
    rows = {}
    with open(ROOT / "data/CSV/train.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[Path(row["ecg_row_record"]).stem] = row
    data = []
    for i, record_id in enumerate(meta["record_ids"]):
        row = rows[str(record_id)]
        data.append(
            {
                "index": i,
                "record_id": str(record_id),
                "patient_id": str(meta["patient_ids"][i]),
                "age": float(row["age"]),
                "sex": int(float(row["gender"])),
                "OMI": int(meta["labels"][i, 0]),
                "AMI": int(meta["labels"][i, 1]),
            }
        )
    frame = pd.DataFrame(data)
    membership = np.full(len(frame), "unused", dtype=object)
    membership[np.asarray(split["train"], dtype=int)] = "train"
    membership[np.asarray(split["val"], dtype=int)] = "validation"
    frame["split"] = membership
    return frame


def point_metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def association_rows(frame):
    rows = []
    for split_name in ("train", "validation"):
        part = frame[frame["split"] == split_name]
        for label in ("OMI", "AMI"):
            y = part[label].to_numpy()
            age = part["age"].to_numpy()
            sex = part["sex"].to_numpy()
            pos_age = age[y == 1]
            neg_age = age[y == 0]
            r_age, p_age = pointbiserialr(y, age)
            _, p_mwu = mannwhitneyu(pos_age, neg_age, alternative="two-sided")
            pooled_sd = np.sqrt((pos_age.var(ddof=1) + neg_age.var(ddof=1)) / 2.0)
            smd = (pos_age.mean() - neg_age.mean()) / pooled_sd if pooled_sd else 0.0
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "feature": "age",
                    "correlation": float(r_age),
                    "p_value": float(p_age),
                    "secondary_test_p": float(p_mwu),
                    "effect_size": float(smd),
                    "positive_summary": f"mean={pos_age.mean():.3f};median={np.median(pos_age):.3f}",
                    "negative_summary": f"mean={neg_age.mean():.3f};median={np.median(neg_age):.3f}",
                    "n_positive": int(y.sum()),
                    "n_negative": int((1 - y).sum()),
                }
            )
            table = np.array(
                [
                    [int(((sex == 1) & (y == 1)).sum()), int(((sex == 1) & (y == 0)).sum())],
                    [int(((sex == 0) & (y == 1)).sum()), int(((sex == 0) & (y == 0)).sum())],
                ]
            )
            odds_ratio, p_fisher = fisher_exact(table)
            r_sex = float(np.corrcoef(sex, y)[0, 1])
            rate_1 = float(y[sex == 1].mean())
            rate_0 = float(y[sex == 0].mean())
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "feature": "sex",
                    "correlation": r_sex,
                    "p_value": float(p_fisher),
                    "secondary_test_p": float("nan"),
                    "effect_size": float(odds_ratio),
                    "positive_summary": f"label_rate_sex1={rate_1:.6f}",
                    "negative_summary": f"label_rate_sex0={rate_0:.6f}",
                    "n_positive": int(y.sum()),
                    "n_negative": int((1 - y).sum()),
                }
            )
    return rows


def logistic_probe_rows(frame):
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "validation"]
    feature_sets = {"age": ["age"], "sex": ["sex"], "age_sex": ["age", "sex"]}
    rows = []
    for label in ("OMI", "AMI"):
        y_train = train[label].to_numpy()
        y_val = val[label].to_numpy()
        for name, columns in feature_sets.items():
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=2026),
            )
            model.fit(train[columns], y_train)
            prob = model.predict_proba(val[columns])[:, 1]
            metrics = point_metrics(y_val, prob)
            rows.append({"label": label, "features": name, **metrics})
    return rows


def write_markdown(assoc, probes, frame):
    lines = [
        "# Age/sex association and clinical-only probe",
        "",
        "All inferential summaries use only the official-training pool and its frozen patient-level train/validation split. The external test was not accessed.",
        "",
        f"Records: train {(frame['split'] == 'train').sum():,}; validation {(frame['split'] == 'validation').sum():,}. Missing age: {int(frame['age'].isna().sum())}; missing sex: {int(frame['sex'].isna().sum())}.",
        "",
        "## Associations",
        "",
        "| Split | Label | Feature | Correlation | p-value | Effect size | Positive group | Negative group |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in assoc:
        lines.append(
            f"| {row['split']} | {row['label']} | {row['feature']} | {row['correlation']:.4f} | "
            f"{row['p_value']:.3g} | {row['effect_size']:.4f} | {row['positive_summary']} | {row['negative_summary']} |"
        )
    lines += [
        "",
        "For age, correlation is point-biserial and effect size is standardized mean difference. For sex, correlation is phi/Pearson and effect size is the Fisher-table odds ratio for sex=1 versus sex=0.",
        "",
        "## Validation logistic probes",
        "",
        "| Label | Features | AUROC | AUPRC | Sens@0.5 | Spec@0.5 | F1@0.5 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in probes:
        lines.append(
            f"| {row['label']} | {row['features']} | {row['auroc']:.4f} | {row['auprc']:.4f} | "
            f"{row['sensitivity']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} |"
        )
    lines += [
        "",
        "These probes quantify marginal predictive association; they do not establish causal effects. Added-value conclusions come from the controlled neural feature ablation.",
    ]
    (OUT / "feature_association_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = aligned_frame()
    assoc = association_rows(frame)
    probes = logistic_probe_rows(frame)
    pd.DataFrame(assoc).to_csv(OUT / "feature_association.csv", index=False)
    pd.DataFrame(probes).to_csv(OUT / "clinical_logistic_probe.csv", index=False)
    write_markdown(assoc, probes, frame)
    print(json.dumps({"association_rows": len(assoc), "probe_rows": len(probes)}, indent=2))


if __name__ == "__main__":
    main()
