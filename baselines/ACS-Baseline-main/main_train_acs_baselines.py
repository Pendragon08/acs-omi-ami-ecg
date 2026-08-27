from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import models


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "ACS" / "preprocessed_100hz" / "normalized" / "train"
DEFAULT_CSV_PATH = BASE_DIR / "data" / "ACS" / "CSV" / "train.csv"
DEFAULT_CHECKPOINT_DIR = BASE_DIR / "checkpoints"
DEFAULT_RESULT_DIR = BASE_DIR / "results"
MODEL_NAMES = (
    "resnet1d_wang",
    "xresnet1d101",
    "inceptiontime",
    "fcn_wang",
    "lstm",
    "lstm_bidir",
    "vit",
    "mobilenetv3_small",
)
TARGETS = ("AMI", "OMI")
EXPECTED_SHAPE = (1000, 12)


class ACSNpyDataset(Dataset):
    def __init__(self, paths: list[Path], labels: np.ndarray):
        self.paths = paths
        self.labels = labels.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = np.load(self.paths[index], allow_pickle=False)
        signal = np.asarray(signal, dtype=np.float32)
        if signal.shape == (EXPECTED_SHAPE[1], EXPECTED_SHAPE[0]):
            signal = signal.T
        if signal.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"{self.paths[index]} 形状应为 {EXPECTED_SHAPE}，实际为 {signal.shape}"
            )
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(np.ascontiguousarray(signal.T))
        y = torch.tensor([self.labels[index]], dtype=torch.float32)
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 ACS train NPY 上分别进行 AMI、OMI 二分类的八种基线对比实验。"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_matched_records(
    data_dir: Path, csv_path: Path
) -> tuple[list[Path], dict[str, np.ndarray]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"训练 NPY 目录不存在：{data_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"训练 CSV 不存在：{csv_path}")

    npy_by_id = {path.stem: path for path in data_dir.glob("*.npy") if path.is_file()}
    rows_by_id: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"ecg_row_record", *TARGETS}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"CSV 缺少列：{', '.join(sorted(missing_columns))}")
        for row in reader:
            record_id = Path((row.get("ecg_row_record") or "").strip()).stem
            if not record_id:
                continue
            if record_id in rows_by_id:
                raise ValueError(f"CSV 中存在重复记录：{record_id}")
            rows_by_id[record_id] = row

    matched_ids = sorted(npy_by_id.keys() & rows_by_id.keys())
    if not matched_ids:
        raise ValueError("NPY 文件与 CSV 的 ecg_row_record 没有任何匹配记录")

    paths = [npy_by_id[record_id] for record_id in matched_ids]
    labels: dict[str, np.ndarray] = {}
    for target in TARGETS:
        try:
            values = np.asarray([int(rows_by_id[rid][target]) for rid in matched_ids], dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{target} 列包含非整数标签") from exc
        if not np.isin(values, [0, 1]).all():
            raise ValueError(f"{target} 列只能包含 0 或 1")
        labels[target] = values

    print(f"train 目录 NPY：{len(npy_by_id)}")
    print(f"train.csv 记录：{len(rows_by_id)}")
    print(f"成功匹配：{len(matched_ids)}")
    print(f"仅存在于 NPY：{len(npy_by_id.keys() - rows_by_id.keys())}")
    print(f"仅存在于 CSV：{len(rows_by_id.keys() - npy_by_id.keys())}")
    for target in TARGETS:
        counts = np.bincount(labels[target], minlength=2)
        print(f"{target} 标签分布：0={counts[0]}，1={counts[1]}")
    return paths, labels


def make_loaders(
    paths: list[Path], labels: np.ndarray, batch_size: int, workers: int, seed: int
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    indices = np.arange(len(paths))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    train_paths = [paths[index] for index in train_indices]
    val_paths = [paths[index] for index in val_indices]
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        ACSNpyDataset(train_paths, train_labels),
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        ACSNpyDataset(val_paths, val_labels), shuffle=False, **common
    )
    split_stats = {
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "train_negative": int((train_labels == 0).sum()),
        "train_positive": int((train_labels == 1).sum()),
        "val_negative": int((val_labels == 0).sum()),
        "val_positive": int((val_labels == 1).sum()),
    }
    return train_loader, val_loader, split_stats


def compute_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    return {
        "acc": float(accuracy_score(targets, predictions)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "auc": float(roc_auc_score(targets, probabilities)),
        "tpr": float(tp / (tp + fn)) if tp + fn else 0.0,
        "tnr": float(tn / (tn + fp)) if tn + fp else 0.0,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                inputs = inputs + torch.randn_like(inputs) * 0.1
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = sum(logits) / len(logits)
            logits = logits.reshape_as(labels)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.shape[0]
            targets.append(labels.detach().cpu().numpy().reshape(-1))
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))

    all_targets = np.concatenate(targets)
    all_probabilities = np.concatenate(probabilities)
    metrics = compute_metrics(all_targets, all_probabilities)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_auc: float,
    model_name: str,
    target: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "global_epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc,
            "model_name": model_name,
            "target": target,
        },
        path,
    )


def train_experiment(
    model_name: str,
    target: str,
    paths: list[Path],
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    setup_seed(args.seed)
    train_loader, val_loader, split_stats = make_loaders(
        paths, labels, args.batch_size, args.workers, args.seed
    )
    model = getattr(models, model_name)(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    experiment = f"acs_{target.lower()}"
    checkpoint_path = args.checkpoint_dir / f"{model_name}_{experiment}_checkpoint_best.pth"
    result_path = args.result_dir / f"{model_name}_{experiment}_result.csv"
    summary_path = args.result_dir / f"{model_name}_{experiment}_best.json"

    print("\n" + "=" * 80)
    print(f"模型：{model_name} | 任务：{target} | 设备：{device}")
    print(f"划分：{split_stats}")
    best_auc = float("-inf")
    best_metrics: dict[str, float | int | str] = {}
    started = time.time()

    with result_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "epoch", "train_loss", "train_acc", "train_f1", "train_auc", "train_tpr", "train_tnr",
            "val_loss", "val_acc", "val_f1", "val_auc", "val_tpr", "val_tnr", "best_val_auc",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.time()
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
            val_metrics = run_epoch(model, val_loader, criterion, device)
            improved = val_metrics["auc"] > best_auc
            if improved:
                best_auc = val_metrics["auc"]
                save_checkpoint(
                    checkpoint_path, model, optimizer, epoch, best_auc, model_name, target
                )
                best_metrics = {
                    "model": model_name,
                    "target": target,
                    "best_epoch": epoch,
                    "best_val_auc": best_auc,
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                    **split_stats,
                }
                summary_path.write_text(
                    json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            writer.writerow(
                {
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                    "best_val_auc": best_auc,
                }
            )
            file.flush()
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss={train_metrics['loss']:.4f} auc={train_metrics['auc']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} auc={val_metrics['auc']:.4f} "
                f"f1={val_metrics['f1']:.4f} | best={best_auc:.4f}" +
                (" [已保存]" if improved else "") +
                f" | {time.time() - epoch_started:.1f}s"
            )

    print(f"实验完成，耗时 {(time.time() - started) / 60:.1f} 分钟")
    print(f"最佳权重：{checkpoint_path}")
    print(f"逐轮结果：{result_path}")
    print(f"最佳结果：{summary_path}")


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.csv_path = args.csv_path.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.result_dir = args.result_dir.expanduser().resolve()
    if args.batch_size < 1 or args.epochs < 1 or args.workers < 0:
        raise ValueError("batch-size、epochs 必须大于 0，workers 不能小于 0")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    paths, labels_by_target = load_matched_records(args.data_dir, args.csv_path)
    print(f"测试目录不会读取或处理：{args.data_dir.parent / 'test'}")

    for target in args.targets:
        for model_name in args.models:
            train_experiment(
                model_name, target, paths, labels_by_target[target], args, device
            )


if __name__ == "__main__":
    main()
