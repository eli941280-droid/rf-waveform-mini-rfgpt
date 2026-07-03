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
from torch.utils.data import DataLoader

from rf_dataset import LABELS, WTRSpectrogramDataset, class_counts, collate_wtr_batch


class SmallRFCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight CNN baseline for Mini-WTR.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-jsonl", type=Path, default=Path("data_all/splits/wtr_train.jsonl"))
    parser.add_argument("--val-jsonl", type=Path, default=Path("data_all/splits/wtr_val.jsonl"))
    parser.add_argument("--test-jsonl", type=Path, default=Path("data_all/splits/wtr_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/python_wtr_cnn"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate an existing checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path for --eval-only. Defaults to output-dir/best_model.pt.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_set = WTRSpectrogramDataset(args.train_jsonl, args.project_root, args.image_size)
    val_set = WTRSpectrogramDataset(args.val_jsonl, args.project_root, args.image_size)
    test_set = WTRSpectrogramDataset(args.test_jsonl, args.project_root, args.image_size)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_wtr_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_wtr_batch,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_wtr_batch,
    )

    model = SmallRFCNN(num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train/val/test samples: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print(f"Class counts train: {class_counts(train_set)}")

    history = []
    checkpoint_path = args.output_dir / "best_model.pt"
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    start_time = time.time()

    if args.eval_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for --eval-only: {checkpoint_path}")
        print(f"Eval-only mode: loading checkpoint {checkpoint_path}")
    else:
        best_val_acc = -1.0
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(epoch_record)

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
                        "labels": LABELS,
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
    test_loss, test_acc, y_true, y_pred, pred_probs = evaluate(model, test_loader, criterion, device)

    elapsed_sec = time.time() - start_time
    confusion = confusion_matrix(y_true, y_pred, len(LABELS))

    report = {
        "model": "SmallRFCNN",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "labels": LABELS,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "elapsed_sec": elapsed_sec,
        "confusion_matrix": confusion.tolist(),
        "history": history,
    }

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    write_predictions(args.output_dir / "test_predictions.csv", test_set, y_true, y_pred, pred_probs)
    plot_confusion(args.output_dir / "confusion_matrix.png", confusion, LABELS)
    if history:
        plot_history(args.output_dir / "training_curve.png", history)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Report: {report_path}")
    print(f"Checkpoint: {checkpoint_path}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true = []
    y_pred = []
    pred_probs = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (pred == labels).sum().item()
        total_count += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        pred_probs.extend(probs.cpu().tolist())

    return (
        total_loss / total_count,
        total_correct / total_count,
        np.array(y_true, dtype=np.int64),
        np.array(y_pred, dtype=np.int64),
        np.array(pred_probs, dtype=np.float32),
    )


def write_predictions(path, dataset, y_true, y_pred, pred_probs):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "image", "true_label", "pred_label", "correct", "confidence"])
        for idx, record in enumerate(dataset.records):
            confidence = float(np.max(pred_probs[idx]))
            writer.writerow(
                [
                    record.get("id", ""),
                    record.get("image", ""),
                    LABELS[int(y_true[idx])],
                    LABELS[int(y_pred[idx])],
                    int(y_true[idx] == y_pred[idx]),
                    f"{confidence:.6f}",
                ]
            )


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_id, pred_id in zip(y_true, y_pred):
        matrix[int(true_id), int(pred_id)] += 1
    return matrix


def plot_confusion(path, matrix, labels):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("WTR CNN Confusion Matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_history(path, history):
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


def choose_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    main()
