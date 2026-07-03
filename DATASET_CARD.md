# Mini RF-GPT / Mini-WTR Synthetic Dataset Card

## Dataset Summary

This dataset is a small synthetic RF spectrogram dataset generated in MATLAB for Mini RF-GPT / Mini-WTR reproduction experiments.

- Total samples: 600 RF spectrogram scenes
- Technologies: 6
- Samples per technology: 100
- Image size: 512 x 512 grayscale PNG
- IQ files: complex baseband `.mat`
- Instruction records: 3300
- WTR benchmark records: 600
- Splits: sample-level stratified 80/10/10 per technology

## Technologies

- 5G NR
- LTE
- UMTS
- WLAN
- DVB-S2
- Bluetooth

## Generated Files

- `data_all/metadata_index.jsonl`
- `data_all/instruction_data.jsonl`
- `data_all/wtr_benchmark.jsonl`
- `data_all/splits/instruction_train.jsonl`
- `data_all/splits/instruction_val.jsonl`
- `data_all/splits/instruction_test.jsonl`
- `data_all/splits/wtr_train.jsonl`
- `data_all/splits/wtr_val.jsonl`
- `data_all/splits/wtr_test.jsonl`
- `data_all/vlm_sft/llava_train.jsonl`
- `data_all/vlm_sft/llava_val.jsonl`
- `data_all/vlm_sft/llava_test.jsonl`

## Generation Pipeline

Waveforms are generated with technology-specific MATLAB generators under `generators/`, followed by lightweight RF impairments and technology-aware visual profiling under `utils/`.

The spectrogram renderer uses STFT-based grayscale images. Metadata are generated directly from the controlled synthetic pipeline.

## Quality Audit

The latest audit is saved at:

- `outputs/dataset_quality_audit.txt`

Current audit status:

- JSONL decode errors: 0
- Duplicate sample IDs: 0
- Missing waveform files: 0
- Missing spectrogram files: 0
- Non-512x512 images: 0
- Split leakage issues: 0
- Nearest-centroid WTR accuracy: 90.00%
- 1-NN cosine WTR accuracy: 88.33%
- Decision: PASS

## Lightweight Baseline

The script `run_wtr_baseline.m` evaluates low-compute visual classifiers over the WTR split.

Current result:

- Selected method: 1-NN cosine (`knn1`)
- Selection rule: validation accuracy with fixed tie-break priority `knn1 > centroid > knn5`
- Test accuracy: 91.67%
- Report: `outputs/wtr_baseline_report.txt`
- Predictions: `outputs/wtr_baseline_predictions.csv`
- Confusion matrix: `outputs/wtr_baseline_confusion.png`

## Generalization Stress Test

The script `main_generate_hard_test_dataset.m` creates three synthetic out-of-distribution WTR test domains under `data_hard/`:

- `shifted_impairment`: keeps the technology-aware visual profile but shifts SNR, frequency offset, and spectrogram dynamic range.
- `weak_profile`: replaces technology-aware shaping with generic time/frequency augmentation.
- `no_profile`: uses raw generator waveforms with only frequency offset and AWGN.

The trained CNN checkpoint `outputs/python_wtr_cnn/best_model.pt` was evaluated with:

```bash
python python/eval_wtr_cnn.py --device auto --checkpoint outputs/python_wtr_cnn/best_model.pt --output-dir outputs/hard_eval
```

Current hard-test results:

- `shifted_impairment`: 73.33% accuracy
- `weak_profile`: 40.00% accuracy
- `no_profile`: 45.00% accuracy

These results indicate substantial dependence on the synthetic technology-aware visual profiles. Same-distribution WTR accuracy should therefore be reported as a pipeline sanity result, not as evidence of robust real-world RF generalization.

## Robust And Held-Out Protocols

The script `build_robust_wtr_splits.m` combines `data_all/` and `data_hard/` into `data_robust/` and stratifies by both domain and technology.

Current robust-split CNN result:

- Train/validation/test records: 798/162/180
- Overall test accuracy: 96.11%
- `in_distribution`: 100.00%
- `shifted_impairment`: 100.00%
- `weak_profile`: 83.33%
- `no_profile`: 93.33%

The script `build_domain_holdout_wtr_splits.m` creates stricter domain-held-out splits. In the current `no_profile` held-out experiment, training uses `in_distribution`, `shifted_impairment`, and `weak_profile`, while all `no_profile` samples are kept only for testing.

Current `no_profile` held-out CNN result:

- Test accuracy: 85.00%
- 5G NR: 100.00%
- LTE: 100.00%
- UMTS: 96.67%
- WLAN: 100.00%
- Bluetooth: 80.00%
- DVB-S2: 33.33%

This indicates that mixed-domain training improves robustness substantially, but profile-free DVB-S2 remains a weak point and should be improved before making stronger claims.

## Six-Technology Five-Task Benchmark

The script `build_sixtech_fivetask_benchmark.m` creates a compact RF-GPT-style classification benchmark from `data_robust/wtr_benchmark.jsonl`.

Current benchmark:

- Samples: 1140
- Tasks per sample: 5
- Total records: 5700
- Train/validation/test records: 3990/810/900
- Split policy: sample-level split inherited from `data_robust/splits/sample_manifest.jsonl`

Tasks:

- `technology_recognition`
- `snr_bucket`
- `time_occupancy`
- `frequency_occupancy`
- `domain_condition`

Files:

- `data_robust/sixtech_fivetask_benchmark.jsonl`
- `data_robust/splits/sixtech_fivetask_train.jsonl`
- `data_robust/splits/sixtech_fivetask_val.jsonl`
- `data_robust/splits/sixtech_fivetask_test.jsonl`
- `data_robust/splits/sixtech_fivetask_task_counts.csv`

See `docs/SIXTECH_FIVETASK_BENCHMARK.md` for label definitions and limitations.

## Recommended Uses

- Mini RF-GPT data pipeline reproduction
- Wireless technology recognition sanity experiments
- Visual instruction tuning smoke tests
- RF spectrogram metadata-to-instruction experiments

## Not Recommended Uses

- Real over-the-air RF performance claims
- Deployment-facing RF classifier benchmarking
- Channel estimation, demodulation, or PHY receiver performance evaluation
- Claims about real-world RF generalization

## Known Limitations

- The dataset is synthetic and should not be described as real RF capture data.
- Technology-aware visual profiles intentionally strengthen class separability for the mini-WTR task.
- Hard-test evaluation shows that CNN accuracy drops sharply when these profiles are weakened or removed.
- Robust and domain-held-out protocols are more informative than the original same-distribution split.
- Current scenes are mostly single-signal scenes; dense multi-signal RF scene reasoning is not covered.
- DVB-S2 may use a fallback waveform generator when official toolbox resources are unavailable.
- Instruction answers are template-generated rather than LLM-diversified dense captions.

## Reproducibility

From the project root:

```matlab
run_next_pipeline
```

This regenerates the six-technology dataset, instruction JSONL, WTR benchmark, splits, similarity diagnostics, and quality audit.
