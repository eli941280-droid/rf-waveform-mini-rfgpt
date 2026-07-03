import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions for the six-technology five-task RF benchmark."
    )
    parser.add_argument(
        "--gold-jsonl",
        type=Path,
        default=Path("data_robust/splits/sixtech_fivetask_test.jsonl"),
    )
    parser.add_argument(
        "--pred-jsonl",
        type=Path,
        required=True,
        help="JSONL with id and prediction fields. Also accepts pred, answer, response, or output.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fivetask_eval"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gold = read_gold(args.gold_jsonl)
    preds = read_predictions(args.pred_jsonl)

    rows = []
    overall = Counter()
    by_task = defaultdict(Counter)

    for record_id, rec in gold.items():
        target = normalize(rec["answer"])
        pred_raw = preds.get(record_id, "")
        pred = normalize(pred_raw)
        correct = int(pred == target)

        overall["total"] += 1
        overall["correct"] += correct
        by_task[rec["task"]]["total"] += 1
        by_task[rec["task"]]["correct"] += correct

        rows.append(
            {
                "id": record_id,
                "task": rec["task"],
                "answer": rec["answer"],
                "prediction": pred_raw,
                "correct": correct,
            }
        )

    report = {
        "gold_jsonl": str(args.gold_jsonl),
        "pred_jsonl": str(args.pred_jsonl),
        "gold_records": len(gold),
        "prediction_records": len(preds),
        "missing_predictions": sum(1 for rid in gold if rid not in preds),
        "overall_accuracy": safe_acc(overall),
        "task_metrics": {
            task: {
                "correct": counts["correct"],
                "total": counts["total"],
                "accuracy": safe_acc(counts),
            }
            for task, counts in sorted(by_task.items())
        },
    }

    report_path = args.output_dir / "report.json"
    detail_path = args.output_dir / "details.csv"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_details(detail_path, rows)

    print(f"Gold records: {len(gold)}")
    print(f"Prediction records: {len(preds)}")
    print(f"Missing predictions: {report['missing_predictions']}")
    print(f"Overall accuracy: {report['overall_accuracy']:.4f}")
    print("Task metrics:")
    for task, item in report["task_metrics"].items():
        print(f"  {task}: {item['accuracy']:.4f} ({item['correct']}/{item['total']})")
    print(f"Report: {report_path}")
    print(f"Details: {detail_path}")


def read_gold(path):
    records = {}
    for line_number, rec in read_jsonl(path):
        record_id = rec.get("id", "")
        if not record_id:
            raise ValueError(f"Missing id in gold file at {path}:{line_number}")
        records[record_id] = {
            "task": rec.get("task", "unknown"),
            "answer": str(rec.get("answer", "")),
        }
    return records


def read_predictions(path):
    predictions = {}
    for line_number, rec in read_jsonl(path):
        record_id = rec.get("id", "")
        if not record_id:
            raise ValueError(f"Missing id in prediction file at {path}:{line_number}")
        predictions[record_id] = extract_prediction(rec)
    return predictions


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def extract_prediction(rec):
    for key in ["prediction", "pred", "answer", "response", "output"]:
        if key in rec and rec[key] is not None:
            return str(rec[key])
    return ""


def normalize(text):
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def safe_acc(counter):
    total = counter["total"]
    return counter["correct"] / total if total else 0.0


def write_details(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "task", "answer", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
