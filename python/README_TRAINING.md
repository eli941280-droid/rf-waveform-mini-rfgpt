# Local PyTorch Training

This folder contains a lightweight local training baseline for the Mini-WTR task.

## Environment

Install dependencies:

```bash
pip install -r python/requirements.txt
```

If you use an RTX 4090/4090 Laptop GPU, install a CUDA-enabled PyTorch build that matches your local CUDA driver.

The training script sets `KMP_DUPLICATE_LIB_OK=TRUE` by default to avoid a common Windows OpenMP duplicate-runtime crash caused by mixed scientific Python packages.

## CNN WTR Baseline

From the project root:

```bash
python python/train_wtr_cnn.py --device auto --epochs 30 --batch-size 32 --image-size 224
```

For an RTX 4090 Laptop GPU with 16 GB VRAM, this command should be comfortable. For a desktop RTX 4090 with 24 GB VRAM, you can try `--batch-size 64`.

Outputs:

- `outputs/python_wtr_cnn/report.json`
- `outputs/python_wtr_cnn/best_model.pt`
- `outputs/python_wtr_cnn/test_predictions.csv`
- `outputs/python_wtr_cnn/confusion_matrix.png`
- `outputs/python_wtr_cnn/training_curve.png`

If training finished but crashed while loading the checkpoint for final evaluation, run:

```bash
python python/train_wtr_cnn.py --device auto --eval-only --checkpoint outputs/python_wtr_cnn/best_model.pt --output-dir outputs/python_wtr_cnn
```

This reuses the saved checkpoint and regenerates `report.json`, `test_predictions.csv`, and `confusion_matrix.png` without retraining.

For a quick smoke test:

```bash
python python/train_wtr_cnn.py --device cpu --epochs 1 --batch-size 16 --image-size 96 --output-dir outputs/python_wtr_cnn_smoke
```

This smoke test has been verified locally and writes a full report/checkpoint/plot set to `outputs/python_wtr_cnn_smoke`.

## Hard-Test Evaluation

After generating the hard-test domains in MATLAB:

```matlab
main_generate_hard_test_dataset
```

evaluate the existing checkpoint without retraining:

```bash
python python/eval_wtr_cnn.py --device auto --checkpoint outputs/python_wtr_cnn/best_model.pt --output-dir outputs/hard_eval
```

Current result on the trained checkpoint:

- `shifted_impairment`: 73.33% accuracy
- `weak_profile`: 40.00% accuracy
- `no_profile`: 45.00% accuracy

This is the important generalization check: high same-distribution accuracy alone is not enough for a real RF generalization claim.

## Robust And Held-Out Training

Build the mixed-domain robust split:

```matlab
build_robust_wtr_splits
```

Train on it:

```bash
python python/train_wtr_cnn.py --device auto --epochs 30 --batch-size 32 --image-size 224 --train-jsonl data_robust/splits/wtr_train.jsonl --val-jsonl data_robust/splits/wtr_val.jsonl --test-jsonl data_robust/splits/wtr_test.jsonl --output-dir outputs/python_wtr_cnn_robust
```

Evaluate by domain:

```bash
python python/eval_wtr_cnn.py --device auto --checkpoint outputs/python_wtr_cnn_robust/best_model.pt --wtr-jsonl data_robust/splits/wtr_test.jsonl --output-dir outputs/robust_eval
```

Current robust result: 96.11% overall test accuracy, with 83.33% on `weak_profile` and 93.33% on `no_profile`.

For stricter domain-held-out splits:

```matlab
build_domain_holdout_wtr_splits
```

Current `no_profile` held-out run:

```bash
python python/train_wtr_cnn.py --device auto --epochs 30 --batch-size 32 --image-size 224 --train-jsonl data_domain_holdout/no_profile/splits/wtr_train.jsonl --val-jsonl data_domain_holdout/no_profile/splits/wtr_val.jsonl --test-jsonl data_domain_holdout/no_profile/splits/wtr_test.jsonl --output-dir outputs/python_wtr_cnn_holdout_no_profile
```

This reaches 85.00% test accuracy on a completely held-out `no_profile` domain. DVB-S2 remains the main failure case.

## Five-Task CNN Baseline

The six-technology five-task benchmark can be trained with a lightweight shared-backbone CNN:

```bash
python python/train_fivetask_cnn.py --device auto --epochs 20 --batch-size 64 --image-size 224
```

Outputs:

- `outputs/python_fivetask_cnn/report.json`
- `outputs/python_fivetask_cnn/best_model.pt`
- `outputs/python_fivetask_cnn/test_predictions.csv`
- `outputs/python_fivetask_cnn/test_predictions.jsonl`
- `outputs/python_fivetask_cnn/training_curve.png`
- `outputs/python_fivetask_cnn/task_accuracy.png`

Score its predictions with:

```bash
python python/eval_fivetask_predictions.py --gold-jsonl data_robust/splits/sixtech_fivetask_test.jsonl --pred-jsonl outputs/python_fivetask_cnn/test_predictions.jsonl --output-dir outputs/fivetask_eval
```

Kaggle deployment notes are in `docs/KAGGLE_TRAINING.md`.

## Notes

- The dataset is synthetic and should not be reported as real over-the-air RF capture data.
- This CNN is a sanity baseline, not the final RF-GPT/VLM model.
- The next VLM step can use `data_all/vlm_sft/llava_train.jsonl`, `llava_val.jsonl`, and `llava_test.jsonl`.
