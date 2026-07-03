# Kaggle Training Guide

This project can be trained on Kaggle, but Kaggle should be used for Python training only.

Do not rely on Kaggle for MATLAB waveform generation. Generate and freeze the dataset locally first, then upload the generated files as a Kaggle Dataset.

## What To Upload

Create a Kaggle Dataset containing at least:

- `data_all/`
- `data_hard/`
- `data_robust/`
- `python/`
- `README.md`
- `DATASET_CARD.md`
- `docs/`

For training only, `data_robust/` and `python/` are enough. Keeping `data_all/` and `data_hard/` is useful for traceability.

## Kaggle Notebook Setup

1. Create a Kaggle Notebook.
2. Attach the uploaded dataset.
3. Enable GPU in Notebook settings.
4. Run commands from `/kaggle/working`.

Assume the attached dataset is available at:

```text
/kaggle/input/rf-waveform-mini
```

If the actual folder name is different, replace it in the commands below.

## Install Dependencies

Kaggle usually includes PyTorch, NumPy, PIL, and Matplotlib. If needed:

```bash
pip install -r /kaggle/input/rf-waveform-mini/python/requirements.txt
```

## Train Wireless Technology Recognition CNN

```bash
python /kaggle/input/rf-waveform-mini/python/train_wtr_cnn.py \
  --device auto \
  --project-root /kaggle/input/rf-waveform-mini \
  --train-jsonl /kaggle/input/rf-waveform-mini/data_robust/splits/wtr_train.jsonl \
  --val-jsonl /kaggle/input/rf-waveform-mini/data_robust/splits/wtr_val.jsonl \
  --test-jsonl /kaggle/input/rf-waveform-mini/data_robust/splits/wtr_test.jsonl \
  --output-dir /kaggle/working/python_wtr_cnn_robust \
  --epochs 30 \
  --batch-size 64 \
  --image-size 224
```

If GPU memory is tight, use `--batch-size 32`.

## Train Six-Technology Five-Task CNN

```bash
python /kaggle/input/rf-waveform-mini/python/train_fivetask_cnn.py \
  --device auto \
  --project-root /kaggle/input/rf-waveform-mini \
  --train-jsonl /kaggle/input/rf-waveform-mini/data_robust/splits/sixtech_fivetask_train.jsonl \
  --val-jsonl /kaggle/input/rf-waveform-mini/data_robust/splits/sixtech_fivetask_val.jsonl \
  --test-jsonl /kaggle/input/rf-waveform-mini/data_robust/splits/sixtech_fivetask_test.jsonl \
  --output-dir /kaggle/working/python_fivetask_cnn \
  --epochs 20 \
  --batch-size 64 \
  --image-size 224
```

The five-task trainer writes:

- `report.json`
- `best_model.pt`
- `test_predictions.csv`
- `test_predictions.jsonl`
- `training_curve.png`
- `task_accuracy.png`

## Score Five-Task Predictions

```bash
python /kaggle/input/rf-waveform-mini/python/eval_fivetask_predictions.py \
  --gold-jsonl /kaggle/input/rf-waveform-mini/data_robust/splits/sixtech_fivetask_test.jsonl \
  --pred-jsonl /kaggle/working/python_fivetask_cnn/test_predictions.jsonl \
  --output-dir /kaggle/working/fivetask_eval
```

## Recommended Deliverables

Download these from Kaggle output:

- `python_wtr_cnn_robust/report.json`
- `python_wtr_cnn_robust/confusion_matrix.png`
- `python_wtr_cnn_robust/training_curve.png`
- `python_fivetask_cnn/report.json`
- `python_fivetask_cnn/task_accuracy.png`
- `python_fivetask_cnn/training_curve.png`
- `fivetask_eval/report.json`

## Notes

- JSONL image paths generated on Windows contain backslashes. The Python dataset loader normalizes these paths for Kaggle/Linux.
- Kaggle outputs should be written under `/kaggle/working`, not under `/kaggle/input`.
- This is a synthetic benchmark. Do not describe the result as real over-the-air RF generalization.
