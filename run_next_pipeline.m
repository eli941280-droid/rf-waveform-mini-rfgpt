clc; clear; close all;

startup;

fprintf("Running mini RF-GPT all-technology pipeline...\n");

main_generate_all_tech_dataset;
build_instruction_jsonl_all;
build_wtr_benchmark_jsonl;
build_dataset_splits;
export_vlm_sft_jsonl;
compare_all_tech_similarity;
audit_dataset_quality;
run_wtr_baseline;

fprintf("\nPipeline complete.\n");
fprintf("Dataset: %s\n", fullfile(pwd, "data_all"));
fprintf("Splits: %s\n", fullfile(pwd, "data_all", "splits"));
fprintf("VLM SFT export: %s\n", fullfile(pwd, "data_all", "vlm_sft"));
fprintf("Similarity report: %s\n", fullfile(pwd, "outputs", "compare_all_tech_similarity.txt"));
fprintf("Quality audit: %s\n", fullfile(pwd, "outputs", "dataset_quality_audit.txt"));
fprintf("WTR baseline: %s\n", fullfile(pwd, "outputs", "wtr_baseline_report.txt"));
