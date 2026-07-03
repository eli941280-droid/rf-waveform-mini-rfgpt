import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


LABELS = ["5G NR", "LTE", "UMTS", "WLAN", "DVB-S2", "Bluetooth"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}


class WTRSpectrogramDataset(Dataset):
    def __init__(self, jsonl_path, project_root=".", image_size=224):
        self.jsonl_path = Path(jsonl_path)
        self.project_root = Path(project_root)
        self.image_size = image_size
        self.records = self._read_jsonl(self.jsonl_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = self.project_root / normalize_relative_path(record["image"])
        image = Image.open(image_path).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        tensor = torch.from_numpy(np.asarray(image, dtype="float32")).unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5

        label = record["answer"]
        if label not in LABEL_TO_ID:
            raise ValueError(f"Unknown label {label!r} in {self.jsonl_path}")

        return {
            "image": tensor,
            "label": torch.tensor(LABEL_TO_ID[label], dtype=torch.long),
            "id": record.get("id", ""),
            "image_path": str(image_path),
            "label_name": label,
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
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        return records


def collate_wtr_batch(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    ids = [item["id"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    label_names = [item["label_name"] for item in batch]
    return {
        "image": images,
        "label": labels,
        "id": ids,
        "image_path": image_paths,
        "label_name": label_names,
    }


def class_counts(dataset):
    counts = {label: 0 for label in LABELS}
    for record in dataset.records:
        counts[record["answer"]] += 1
    return counts


def normalize_relative_path(path):
    return Path(str(path).replace("\\", "/"))
