# Mini RF-GPT Reproduction Summary

## Project Positioning

This repository provides a small-scale, synthetic reproduction pipeline inspired by RF-GPT / wireless technology recognition workflows.

It focuses on a practical and reproducible subset:

1. MATLAB-based RF waveform generation.
2. RF spectrogram rendering.
3. Metadata and instruction-style JSONL construction.
4. Wireless technology recognition benchmarks.
5. Robustness and domain-shift evaluation.
6. Lightweight CNN baselines for quick local or Kaggle training.

This is not a full reproduction of the original RF-GPT model training. It is a compact data factory and benchmark framework for small-scale RF spectrogram experiments.

## Supported Signal Types

- 5G NR
- LTE
- UMTS / WCDMA
- WLAN / Wi-Fi
- DVB-S2
- Bluetooth LE

## Main Generated Benchmarks

### Mini-WTR Benchmark

The Mini-WTR benchmark is a six-class wireless technology recognition task.

- Base samples: 600
- Six classes, 100 samples per class
- Sample-level train/validation/test splits
- Robust mixed-domain split: 1140 records

### Six-Technology Five-Task Benchmark

The five-task benchmark contains five classification-style visual QA tasks for each RF spectrogram.

- Samples: 1140
- Tasks per sample: 5
- Total records: 5700
- Train/validation/test records: 3990 / 810 / 900

Tasks:

- `technology_recognition`
- `snr_bucket`
- `time_occupancy`
- `frequency_occupancy`
- `domain_condition`

The labels are synthetic controlled labels generated from the waveform and augmentation metadata. They are intended for small-scale RF visual instruction experiments, not real over-the-air measurement claims.

## Current Results

### Same-Distribution WTR CNN

The first CNN baseline reaches very high validation/test accuracy on the same synthetic distribution. This validates that the pipeline is learnable, but it is not enough to claim general RF robustness.

### Hard-Test Evaluation

A checkpoint trained on the original distribution was evaluated on three hard-test domains:

| Test domain | Accuracy |
|---|---:|
| `shifted_impairment` | 73.33% |
| `weak_profile` | 40.00% |
| `no_profile` | 45.00% |

This shows clear dependence on synthetic visual profiles.

### Robust Mixed-Domain CNN

Training with mixed domains improves robustness:

| Metric | Accuracy |
|---|---:|
| Overall robust test | 96.11% |
| `in_distribution` | 100.00% |
| `shifted_impairment` | 100.00% |
| `weak_profile` | 83.33% |
| `no_profile` | 93.33% |

### Domain-Held-Out Evaluation

In the `no_profile` held-out setting, the model is trained without any `no_profile` samples and tested entirely on `no_profile`:

| Class | Accuracy |
|---|---:|
| 5G NR | 100.00% |
| LTE | 100.00% |
| UMTS | 96.67% |
| WLAN | 100.00% |
| Bluetooth | 80.00% |
| DVB-S2 | 33.33% |
| Overall | 85.00% |

DVB-S2 remains the main weakness in the current synthetic pipeline.

## Reproduction Commands

Generate the main dataset:

```matlab
run_next_pipeline
```

Generate hard-test domains:

```matlab
main_generate_hard_test_dataset
```

Build robust and domain-held-out splits:

```matlab
build_robust_wtr_splits
build_domain_holdout_wtr_splits
```

Build the six-technology five-task benchmark:

```matlab
build_sixtech_fivetask_benchmark
```

Train the robust WTR CNN:

```bash
python python/train_wtr_cnn.py --device auto --epochs 30 --batch-size 32 --image-size 224 --train-jsonl data_robust/splits/wtr_train.jsonl --val-jsonl data_robust/splits/wtr_val.jsonl --test-jsonl data_robust/splits/wtr_test.jsonl --output-dir outputs/python_wtr_cnn_robust
```

Train the five-task CNN:

```bash
python python/train_fivetask_cnn.py --device auto --epochs 20 --batch-size 64 --image-size 224
```

## Kaggle Use

Kaggle should be used for Python training only. Generate datasets locally with MATLAB first, upload `data_robust/` and `python/` as a Kaggle Dataset, then train from a Kaggle Notebook.

See:

- `docs/KAGGLE_TRAINING.md`

## Limitations

- The data are synthetic, not real over-the-air captures.
- Some tasks use controlled metadata labels rather than purely human-observable image labels.
- The pipeline is designed for small-scale reproduction and sanity experiments.
- Current single-signal scenes do not cover dense RF scene reasoning.
- DVB-S2 profile-free robustness needs further improvement.
