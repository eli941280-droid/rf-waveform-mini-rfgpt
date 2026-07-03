import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_wtr_cnn import choose_device, conv_block, load_checkpoint


TASK_LABELS = {
    "technology_recognition": ["5G NR", "LTE", "UMTS", "WLAN", "DVB-S2", "Bluetooth"],
    "snr_bucket": ["low", "medium", "high", "unknown"],
    "time_occupancy": ["full", "single_burst", "double_burst", "periodic_burst", "no_gating", "unknown"],
    "frequency_occupancy": [
        "wideband",
        "moderate_band",
        "narrowband",
        "low_shifted",
        "high_shifted",
        "two_subbands",
        "frequency_hopping",
        "full_spectrum",
        "unknown",
    ],
    "domain_condition": ["in_distribution", "shifted_impairment", "weak_profile", "no_profile", "unknown"],
}
TASKS = list(TASK_LABELS.keys())
TASK_TO_ID = {task: idx for idx, task in enumerate(TASKS)}
LABEL_TO_ID = {task: {label: idx for idx, label in enumerate(labels)} for task, labels in TASK_LABELS.items()}


class FiveTaskSpectrogramDataset(Dataset):
    def __init__(self, jsonl_path, project_root=".", image_size=224):
        self.jsonl_path = Path(jsonl_path)
        self.project_root = Path(project_root)
        self.image_size = image_size
        self.records = self._read_jsonl(self.jsonl_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        task = record["task"]
        answer = record["answer"]

        image_path = self.project_root / normalize_relative_path(record["image"])
        image = Image.open(image_path).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        tensor = torch.from_numpy(np.asarray(image, dtype="float32")).unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5

        return {
            "image": tensor,
            "task_id": torch.tensor(TASK_TO_ID[task], dtype=torch.long),
            "label": torch.tensor(LABEL_TO_ID[task][answer], dtype=torch.long),
            "id": record.get("id", ""),
            "task": task,
            "answer": answer,
            "image_path": str(image_path),
        }

    @staticmethod
    def _read_jsonl(path):
        records = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc

                task = record.get("task", "")
                answer = record.get("answer", "")
                if task not in TASK_LABELS:
                    raise ValueError(f"Unknown task {task!r} at {path}:{line_number}")
                if answer not in LABEL_TO_ID[task]:
                    raise ValueError(f"Unknown answer {answer!r} for task {task!r} at {path}:{line_number}")
                records.append(record)
        return records


class FiveTaskRFCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.20),
        )
        self.heads = nn.ModuleDict(
            {task: nn.Linear(192, len(labels)) for task, labels in TASK_LABELS.items()}
        )

    def forward(self, images):
        features = self.features(images)
        return {task: head(features) for task, head in self.heads.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight multi-task CNN for the six-tech five-task benchmark.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-jsonl", type=Path, default=Path("data_robust/splits/sixtech_fivetask_train.jsonl"))
    parser.add_argument("--val-jsonl", type=Path, default=Path("data_robust/splits/sixtech_fivetask_val.jsonl"))
    parser.add_argument("--test-jsonl", type=Path, default=Path("data_robust/splits/sixtech_fivetask_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/python_fivetask_cnn"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    train_set = FiveTaskSpectrogramDataset(args.train_jsonl, args.project_root, args.image_size)
    val_set = FiveTaskSpectrogramDataset(args.val_jsonl, args.project_root, args.image_size)
    test_set = FiveTaskSpectrogramDataset(args.test_jsonl, args.project_root, args.image_size)

    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers, device)
    test_loader = make_loader(test_set, args.batch_size, False, args.num_workers, device)

    model = FiveTaskRFCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_path = args.checkpoint or (args.output_dir / "best_model.pt")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train/val/test records: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print(f"Train task counts: {task_counts(train_set)}")

    history = []
    start_time = time.time()

    if args.eval_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Eval-only mode: loading checkpoint {checkpoint_path}")
    else:
        best_val_acc = -1.0
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = run_epoch(model, train_loader, optimizer, device)
            val_loss, val_metrics, _, _ = evaluate(model, val_loader, device)
            scheduler.step()

            train_acc = train_metrics["overall"]["accuracy"]
            val_acc = val_metrics["overall"]["accuracy"]
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "task_labels": TASK_LABELS,
                        "args": serializable_args(args),
                        "best_epoch": best_epoch,
                        "best_val_acc": best_val_acc,
                    },
                    checkpoint_path,
                )

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"])
    best_epoch = int(checkpoint.get("best_epoch", 0))
    best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
    test_loss, test_metrics, predictions, pred_jsonl_records = evaluate(model, test_loader, device)

    elapsed_sec = time.time() - start_time
    report = {
        "model": "FiveTaskRFCNN",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "tasks": TASKS,
        "task_labels": TASK_LABELS,
        "train_records": len(train_set),
        "val_records": len(val_set),
        "test_records": len(test_set),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "elapsed_sec": elapsed_sec,
        "history": history,
    }

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_predictions_csv(args.output_dir / "test_predictions.csv", predictions)
    write_predictions_jsonl(args.output_dir / "test_predictions.jsonl", pred_jsonl_records)
    plot_history(args.output_dir / "training_curve.png", history)
    plot_task_accuracy(args.output_dir / "task_accuracy.png", test_metrics)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Test overall accuracy: {test_metrics['overall']['accuracy']:.4f}")
    for task in TASKS:
        print(f"  {task}: {test_metrics[task]['accuracy']:.4f}")
    print(f"Report: {report_path}")
    print(f"Checkpoint: {checkpoint_path}")


def make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )


def collate_batch(batch):
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "task_id": torch.stack([item["task_id"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "id": [item["id"] for item in batch],
        "task": [item["task"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }


def run_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    metric_counts = init_metric_counts()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        task_ids = batch["task_id"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits_by_task = model(images)
        loss = multitask_loss(logits_by_task, task_ids, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        update_metrics(metric_counts, logits_by_task, task_ids, labels)

    return total_loss / total_count, finalize_metrics(metric_counts)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    metric_counts = init_metric_counts()
    predictions = []
    pred_jsonl_records = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        task_ids = batch["task_id"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits_by_task = model(images)
        loss = multitask_loss(logits_by_task, task_ids, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        update_metrics(metric_counts, logits_by_task, task_ids, labels)

        for row in predictions_from_batch(batch, logits_by_task, task_ids, labels):
            predictions.append(row)
            pred_jsonl_records.append({"id": row["id"], "prediction": row["prediction"]})

    return total_loss / total_count, finalize_metrics(metric_counts), predictions, pred_jsonl_records


def multitask_loss(logits_by_task, task_ids, labels):
    losses = []
    weights = []
    for task, task_idx in TASK_TO_ID.items():
        mask = task_ids == task_idx
        if not torch.any(mask):
            continue
        loss = nn.functional.cross_entropy(logits_by_task[task][mask], labels[mask])
        losses.append(loss)
        weights.append(mask.sum().float())

    if not losses:
        raise RuntimeError("Batch contains no recognized tasks.")

    weights = torch.stack(weights)
    weighted_losses = torch.stack(losses) * (weights / weights.sum())
    return weighted_losses.sum()


def update_metrics(metric_counts, logits_by_task, task_ids, labels):
    for task, task_idx in TASK_TO_ID.items():
        mask = task_ids == task_idx
        if not torch.any(mask):
            continue
        pred = logits_by_task[task][mask].argmax(dim=1)
        target = labels[mask]
        correct = int((pred == target).sum().item())
        total = int(target.numel())
        metric_counts[task]["correct"] += correct
        metric_counts[task]["total"] += total
        metric_counts["overall"]["correct"] += correct
        metric_counts["overall"]["total"] += total


def predictions_from_batch(batch, logits_by_task, task_ids, labels):
    rows = []
    task_ids_cpu = task_ids.cpu().tolist()
    labels_cpu = labels.cpu().tolist()
    for idx, task_idx in enumerate(task_ids_cpu):
        task = TASKS[int(task_idx)]
        logits = logits_by_task[task][idx]
        probs = torch.softmax(logits, dim=0)
        pred_id = int(torch.argmax(probs).item())
        label_id = int(labels_cpu[idx])
        rows.append(
            {
                "id": batch["id"][idx],
                "task": task,
                "answer": TASK_LABELS[task][label_id],
                "prediction": TASK_LABELS[task][pred_id],
                "correct": int(pred_id == label_id),
                "confidence": float(probs[pred_id].item()),
            }
        )
    return rows


def init_metric_counts():
    counts = {"overall": {"correct": 0, "total": 0}}
    for task in TASKS:
        counts[task] = {"correct": 0, "total": 0}
    return counts


def finalize_metrics(counts):
    metrics = {}
    for key, item in counts.items():
        total = item["total"]
        correct = item["correct"]
        metrics[key] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }
    return metrics


def task_counts(dataset):
    counts = {task: 0 for task in TASKS}
    for record in dataset.records:
        counts[record["task"]] += 1
    return counts


def write_predictions_csv(path, predictions):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "task", "answer", "prediction", "correct", "confidence"],
        )
        writer.writeheader()
        writer.writerows(predictions)


def write_predictions_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def plot_history(path, history):
    if not history:
        return
    epochs = [item["epoch"] for item in history]
    train_acc = [item["train_acc"] for item in history]
    val_acc = [item["val_acc"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_acc, label="train")
    axes[0].plot(epochs, val_acc, label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, train_loss, label="train")
    axes[1].plot(epochs, val_loss, label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_task_accuracy(path, metrics):
    tasks = TASKS
    values = [metrics[task]["accuracy"] for task in tasks]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(tasks, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Five-Task Test Accuracy")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def normalize_relative_path(path):
    return Path(str(path).replace("\\", "/"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def serializable_args(args):
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


if __name__ == "__main__":
    main()
