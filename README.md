# RF Waveform Mini RF-GPT Data Factory

This MATLAB project builds a small synthetic RF spectrogram dataset aligned with the RF-GPT / WTR workflow.

## Project Layout

- `generators/` - waveform generators for 5G NR, LTE, UMTS, WLAN, DVB-S2, and Bluetooth.
- `utils/` - shared IQ augmentation and spectrogram rendering utilities.
- `scripts/` - dataset generation, instruction JSONL construction, validation, and similarity diagnostics.
- `data_all/` - six-technology dataset, metadata, instruction data, and WTR benchmark.
- `data/` - legacy 5G NR-only dataset.
- `data_wlan/` - legacy WLAN-only dataset.
- `outputs/` - diagnostic reports, plots, and extracted paper notes.
- `docs/` - source paper PDF.

## Main Pipeline

From the project root, run:

```matlab
run_next_pipeline
```

This executes:

```matlab
main_generate_all_tech_dataset
build_instruction_jsonl_all
build_wtr_benchmark_jsonl
build_dataset_splits
export_vlm_sft_jsonl
compare_all_tech_similarity
audit_dataset_quality
run_wtr_baseline
```

Expected outputs:

- `data_all/metadata_index.jsonl`
- `data_all/instruction_data.jsonl`
- `data_all/wtr_benchmark.jsonl`
- `data_all/splits/instruction_train.jsonl`
- `data_all/splits/instruction_val.jsonl`
- `data_all/splits/instruction_test.jsonl`
- `data_all/splits/wtr_test.jsonl`
- `data_all/vlm_sft/llava_train.jsonl`
- `data_all/vlm_sft/llava_val.jsonl`
- `data_all/vlm_sft/llava_test.jsonl`
- `outputs/compare_all_tech_similarity.txt`
- `outputs/compare_all_tech_similarity_heatmap.png`
- `outputs/dataset_quality_audit.txt`
- `outputs/wtr_baseline_report.txt`
- `outputs/wtr_baseline_predictions.csv`
- `outputs/wtr_baseline_confusion.png`

The split files are created at the sample level and stratified by technology, so instructions from the same spectrogram do not leak across train, validation, and test sets.

See `DATASET_CARD.md` for dataset scope, quality status, limitations, and recommended use.

## Quality Gate

Run:

```matlab
audit_dataset_quality
```

The audit checks JSONL integrity, file presence, image size/non-blank status, metadata ranges, split leakage, task counts, and a simple visual WTR baseline. The current dataset passes the audit with 600 valid samples, balanced 80/10/10 splits per technology, nearest-centroid WTR accuracy of 90.00%, and 1-NN cosine WTR accuracy of 88.33%.

## Lightweight WTR Baseline

Run:

```matlab
run_wtr_baseline
```

The baseline compares nearest-centroid, 1-NN, and 5-NN cosine classifiers over spectrogram features. The current selected baseline is `knn1`, chosen by validation accuracy with a fixed tie-break, with 91.67% test accuracy.

## Generalization Stress Test

The ordinary train/validation/test split is same-distribution synthetic data. To check whether a model has learned robust RF structure or mostly the synthetic visual profile, generate hard-test domains:

```matlab
main_generate_hard_test_dataset
```

This writes:

- `data_hard/shifted_impairment/wtr_benchmark.jsonl`
- `data_hard/weak_profile/wtr_benchmark.jsonl`
- `data_hard/no_profile/wtr_benchmark.jsonl`

Evaluate the trained CNN checkpoint without retraining:

```bash
python python/eval_wtr_cnn.py --device auto --checkpoint outputs/python_wtr_cnn/best_model.pt --output-dir outputs/hard_eval
```

Current hard-test result:

- `shifted_impairment`: 73.33% accuracy
- `weak_profile`: 40.00% accuracy
- `no_profile`: 45.00% accuracy

Interpretation: the current CNN learns the synthetic visual profile strongly. The high same-distribution accuracy is useful as a pipeline sanity check, but it should not be claimed as real RF generalization.

For a stronger training protocol, build a mixed-domain robust split:

```matlab
build_robust_wtr_splits
```

Then train:

```bash
python python/train_wtr_cnn.py --device auto --epochs 30 --batch-size 32 --image-size 224 --train-jsonl data_robust/splits/wtr_train.jsonl --val-jsonl data_robust/splits/wtr_val.jsonl --test-jsonl data_robust/splits/wtr_test.jsonl --output-dir outputs/python_wtr_cnn_robust
```

Current robust-split CNN result:

- Overall test accuracy: 96.11%
- `in_distribution`: 100.00%
- `shifted_impairment`: 100.00%
- `weak_profile`: 83.33%
- `no_profile`: 93.33%

For a stricter domain-held-out protocol:

```matlab
build_domain_holdout_wtr_splits
```

The current `no_profile` held-out experiment trains without any `no_profile` samples and tests entirely on `no_profile`, reaching 85.00% accuracy. DVB-S2 remains weak in this setting, with 33.33% class accuracy.

## Six-Technology Five-Task Benchmark

For a compact RF-GPT-style benchmark over all six signal types, run:

```matlab
build_sixtech_fivetask_benchmark
```

This creates a classification-style visual QA benchmark with five tasks per spectrogram:

- `technology_recognition`
- `snr_bucket`
- `time_occupancy`
- `frequency_occupancy`
- `domain_condition`

Current output:

- Samples: 1140
- Benchmark records: 5700
- Train/val/test records: 3990/810/900
- Output: `data_robust/sixtech_fivetask_benchmark.jsonl`
- Splits: `data_robust/splits/sixtech_fivetask_*.jsonl`

See `docs/SIXTECH_FIVETASK_BENCHMARK.md` for task labels and limitations.

Prediction files for this benchmark can be scored with:

```bash
python python/eval_fivetask_predictions.py --gold-jsonl data_robust/splits/sixtech_fivetask_test.jsonl --pred-jsonl path/to/predictions.jsonl --output-dir outputs/fivetask_eval
```

Train the lightweight five-task CNN baseline with:

```bash
python python/train_fivetask_cnn.py --device auto --epochs 20 --batch-size 64 --image-size 224
```

For Kaggle training instructions, see `docs/KAGGLE_TRAINING.md`.

## Local PyTorch Training

For a laptop or desktop RTX 4090, start with the lightweight CNN WTR trainer:

```bash
python python/train_wtr_cnn.py --device auto --epochs 30 --batch-size 32 --image-size 224
```

Smoke test:

```bash
python python/train_wtr_cnn.py --device cpu --epochs 1 --batch-size 16 --image-size 96 --output-dir outputs/python_wtr_cnn_smoke
```

See `python/README_TRAINING.md` for details.

## Path Setup

`startup.m` adds `scripts/`, `generators/`, and `utils/` to the MATLAB path when MATLAB starts from this project root. It can also be run manually:

```matlab
startup
```
