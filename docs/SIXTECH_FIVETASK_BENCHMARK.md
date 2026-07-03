# Six-Technology Five-Task RF Benchmark

This benchmark is a small synthetic RF spectrogram benchmark built from the robust Mini-WTR dataset.

## Scope

- Technologies: 6
- Samples: 1140 RF spectrogram scenes
- Tasks per sample: 5
- Total benchmark records: 5700
- Split policy: sample-level split inherited from `data_robust/splits/sample_manifest.jsonl`

## Technologies

- 5G NR
- LTE
- UMTS
- WLAN
- DVB-S2
- Bluetooth

## Tasks

1. `technology_recognition`
   - Question: Which wireless technology is shown?
   - Labels: `5G NR`, `LTE`, `UMTS`, `WLAN`, `DVB-S2`, `Bluetooth`

2. `snr_bucket`
   - Question: What is the simulated SNR level bucket?
   - Labels: `low`, `medium`, `high`, `unknown`
   - Note: This is a synthetic metadata-supported bucket, not exact SNR extraction.

3. `time_occupancy`
   - Question: What is the approximate time occupancy pattern?
   - Labels: `full`, `single_burst`, `double_burst`, `periodic_burst`, `no_gating`, `unknown`

4. `frequency_occupancy`
   - Question: What is the approximate frequency occupancy pattern?
   - Labels: `wideband`, `moderate_band`, `narrowband`, `low_shifted`, `high_shifted`, `two_subbands`, `frequency_hopping`, `full_spectrum`, `unknown`

5. `domain_condition`
   - Question: Which synthetic domain condition does this spectrogram belong to?
   - Labels: `in_distribution`, `shifted_impairment`, `weak_profile`, `no_profile`, `unknown`
   - Note: This is a controlled synthetic-domain label for robustness analysis.

## Files

- `data_robust/sixtech_fivetask_benchmark.jsonl`
- `data_robust/splits/sixtech_fivetask_train.jsonl`
- `data_robust/splits/sixtech_fivetask_val.jsonl`
- `data_robust/splits/sixtech_fivetask_test.jsonl`
- `data_robust/splits/sixtech_fivetask_task_counts.csv`
- `data_robust/splits/sixtech_fivetask_summary.json`

## Current Counts

| Split | Records | Per Task |
|---|---:|---:|
| Train | 3990 | 798 |
| Val | 810 | 162 |
| Test | 900 | 180 |
| All | 5700 | 1140 |

## Build Command

From the MATLAB project root:

```matlab
build_sixtech_fivetask_benchmark
```

## Prediction Evaluation

Model predictions can be evaluated with:

```bash
python python/eval_fivetask_predictions.py --gold-jsonl data_robust/splits/sixtech_fivetask_test.jsonl --pred-jsonl path/to/predictions.jsonl --output-dir outputs/fivetask_eval
```

The prediction JSONL should contain one record per benchmark item:

```json
{"id":"5g_nr_000001_technology_recognition","prediction":"5G NR"}
```

Accepted prediction fields are `prediction`, `pred`, `answer`, `response`, or `output`.

## Limitations

- The benchmark is synthetic and should not be described as real over-the-air RF capture data.
- Some tasks, especially `snr_bucket` and `domain_condition`, are controlled-label tasks rather than purely visual human-observable tasks.
- The benchmark is intended for small-scale RF-GPT / RF visual instruction reproduction, sanity testing, and ablation studies.
