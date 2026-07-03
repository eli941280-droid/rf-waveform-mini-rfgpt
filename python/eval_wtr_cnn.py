import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rf_dataset import LABELS, WTRSpectrogramDataset, class_counts, collate_wtr_batch
from train_wtr_cnn import SmallRFCNN, choose_device, load_checkpoint, plot_confusion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Mini-WTR CNN checkpoint on arbitrary WTR JSONL files."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/python_wtr_cnn/best_model.pt"),
        help="Checkpoint produced by python/train_wtr_cnn.py.",
    )
    parser.add_argument(
        "--wtr-jsonl",
        type=Path,
        nargs="+",
        default=[
            Path("data_hard/shifted_impairment/wtr_benchmark.jsonl"),
            Path("data_hard/weak_profile/wtr_benchmark.jsonl"),
            Path("data_hard/no_profile/wtr_benchmark.jsonl"),
        ],
        help="One or more WTR benchmark JSONL files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hard_eval"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    checkpoint_labels = checkpoint.get("labels", LABELS)
    if list(checkpoint_labels) != list(LABELS):
        raise ValueError(
            f"Checkpoint labels do not match current labels. "
            f"checkpoint={checkpoint_labels}, current={LABELS}"
        )

    model = SmallRFCNN(num_classes=len(LABELS)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss()

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {args.checkpoint}")

    summary = []
    for jsonl_path in args.wtr_jsonl:
        if not jsonl_path.exists():
            print(f"\nSkipping missing WTR JSONL: {jsonl_path}")
            continue

        domain_name = infer_domain_name(jsonl_path)
        domain_dir = args.output_dir / sanitize_name(domain_name)
        domain_dir.mkdir(parents=True, exist_ok=True)

        dataset = WTRSpectrogramDataset(jsonl_path, args.project_root, args.image_size)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_wtr_batch,
        )

        loss, acc, y_true, y_pred, pred_probs = evaluate(model, loader, criterion, device)
        confusion = confusion_matrix(y_true, y_pred, len(LABELS))
        domain_metrics = grouped_accuracy(dataset.records, y_true, y_pred, group_kind="domain")
        label_metrics = grouped_accuracy(dataset.records, y_true, y_pred, group_kind="answer")

        report = {
            "domain": domain_name,
            "wtr_jsonl": str(jsonl_path),
            "checkpoint": str(args.checkpoint),
            "samples": len(dataset),
            "labels": LABELS,
            "class_counts": class_counts(dataset),
            "loss": loss,
            "accuracy": acc,
            "confusion_matrix": confusion.tolist(),
            "domain_metrics": domain_metrics,
            "label_metrics": label_metrics,
        }

        report_path = domain_dir / "report.json"
        predictions_path = domain_dir / "predictions.csv"
        confusion_path = domain_dir / "confusion_matrix.png"
        domain_metrics_path = domain_dir / "domain_metrics.csv"
        label_metrics_path = domain_dir / "label_metrics.csv"

        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_predictions(predictions_path, dataset, y_true, y_pred, pred_probs)
        write_group_metrics(domain_metrics_path, domain_metrics)
        write_group_metrics(label_metrics_path, label_metrics)
        plot_confusion(confusion_path, confusion, LABELS)

        summary.append(
            {
                "domain": domain_name,
                "wtr_jsonl": str(jsonl_path),
                "samples": len(dataset),
                "loss": loss,
                "accuracy": acc,
                "report": str(report_path),
                "predictions": str(predictions_path),
                "confusion_matrix": str(confusion_path),
            }
        )

        print(f"\nDomain: {domain_name}")
        print(f"Samples: {len(dataset)}")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Report: {report_path}")
        print("Domain accuracy:")
        for item in domain_metrics:
            print(f"  {item['group']}: {item['accuracy']:.4f} ({item['correct']}/{item['samples']})")

    write_summary(args.output_dir, summary)
    print_diagnosis(summary)


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

    if total_count == 0:
        raise ValueError("Evaluation set is empty.")

    return (
        total_loss / total_count,
        total_correct / total_count,
        np.array(y_true, dtype=np.int64),
        np.array(y_pred, dtype=np.int64),
        np.array(pred_probs, dtype=np.float32),
    )


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_id, pred_id in zip(y_true, y_pred):
        matrix[int(true_id), int(pred_id)] += 1
    return matrix


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


def grouped_accuracy(records, y_true, y_pred, group_kind):
    groups = {}
    for idx, record in enumerate(records):
        group = get_group_name(record, group_kind)
        if group not in groups:
            groups[group] = {"group": group, "samples": 0, "correct": 0, "accuracy": 0.0}
        groups[group]["samples"] += 1
        groups[group]["correct"] += int(int(y_true[idx]) == int(y_pred[idx]))

    metrics = []
    for group in sorted(groups):
        item = groups[group]
        item["accuracy"] = item["correct"] / item["samples"] if item["samples"] else 0.0
        metrics.append(item)
    return metrics


def get_group_name(record, group_kind):
    if group_kind == "answer":
        return str(record.get("answer", "unknown"))

    if group_kind == "domain":
        if "domain" in record and record["domain"]:
            return str(record["domain"])
        metadata = record.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("domain"):
            return str(metadata["domain"])
        return "unknown"

    return "unknown"


def write_group_metrics(path, metrics):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "samples", "correct", "accuracy"])
        for item in metrics:
            writer.writerow(
                [
                    item["group"],
                    item["samples"],
                    item["correct"],
                    f"{item['accuracy']:.6f}",
                ]
            )


def infer_domain_name(jsonl_path):
    if jsonl_path.name == "wtr_benchmark.jsonl":
        return jsonl_path.parent.name
    return jsonl_path.stem


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "domain"


def write_summary(output_dir, summary):
    summary_json = output_dir / "summary.json"
    summary_txt = output_dir / "summary.txt"

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["Mini-WTR CNN hard-test evaluation summary", ""]
    for item in summary:
        lines.append(
            f"{item['domain']}: samples={item['samples']}, "
            f"loss={item['loss']:.4f}, accuracy={item['accuracy']:.4f}"
        )
    if not summary:
        lines.append("No domains were evaluated.")

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary: {summary_txt}")


def print_diagnosis(summary):
    if not summary:
        print("\nNo hard-test result available.")
        return

    print("\nDiagnostic interpretation:")
    for item in summary:
        acc = item["accuracy"]
        domain = item["domain"]
        if acc >= 0.85:
            verdict = "robust on this synthetic shift"
        elif acc >= 0.60:
            verdict = "partially robust, but generalization is clearly weaker"
        else:
            verdict = "large generalization drop; likely template dependence"
        print(f"- {domain}: {verdict} (acc={acc:.4f})")


if __name__ == "__main__":
    main()
