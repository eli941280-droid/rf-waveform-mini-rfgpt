clc; clear; close all;

inputPath = fullfile("data_robust", "wtr_benchmark.jsonl");
manifestPath = fullfile("data_robust", "splits", "sample_manifest.jsonl");
outputRoot = "data_robust";
splitDir = fullfile(outputRoot, "splits");

outputPath = fullfile(outputRoot, "sixtech_fivetask_benchmark.jsonl");
trainPath = fullfile(splitDir, "sixtech_fivetask_train.jsonl");
valPath = fullfile(splitDir, "sixtech_fivetask_val.jsonl");
testPath = fullfile(splitDir, "sixtech_fivetask_test.jsonl");
summaryPath = fullfile(splitDir, "sixtech_fivetask_summary.json");
taskCountPath = fullfile(splitDir, "sixtech_fivetask_task_counts.csv");

if ~exist(inputPath, "file")
    error("Input robust WTR benchmark not found: %s. Run build_robust_wtr_splits first.", inputPath);
end
if ~exist(manifestPath, "file")
    error("Sample manifest not found: %s. Run build_robust_wtr_splits first.", manifestPath);
end
if ~exist(splitDir, "dir")
    mkdir(splitDir);
end

splitMap = read_split_manifest(manifestPath);

fidIn = fopen(inputPath, "r");
if fidIn < 0
    error("Cannot open input JSONL: %s", inputPath);
end

fidAll = fopen(outputPath, "w");
if fidAll < 0
    fclose(fidIn);
    error("Cannot open output JSONL: %s", outputPath);
end

fidTrain = fopen(trainPath, "w");
fidVal = fopen(valPath, "w");
fidTest = fopen(testPath, "w");
if fidTrain < 0 || fidVal < 0 || fidTest < 0
    close_if_open(fidIn);
    close_if_open(fidAll);
    close_if_open(fidTrain);
    close_if_open(fidVal);
    close_if_open(fidTest);
    error("Cannot open one or more split output files.");
end

cleanupObj = onCleanup(@() cleanup_files(fidIn, fidAll, fidTrain, fidVal, fidTest)); %#ok<NASGU>

sampleCount = 0;
recordCount = 0;
lineNumber = 0;
taskCounts = init_task_count_struct();
splitCounts = init_split_count_struct();

while true
    line = fgetl(fidIn);
    if ~ischar(line)
        break;
    end

    lineNumber = lineNumber + 1;
    line = strtrim(line);
    if isempty(line)
        continue;
    end

    try
        baseRecord = jsondecode(line);
    catch ME
        warning("Skipping input line %d due to JSON parse error: %s", lineNumber, ME.message);
        continue;
    end

    sampleCount = sampleCount + 1;
    sampleId = get_sample_id(baseRecord, sampleCount);
    splitName = get_split(splitMap, sampleId);
    records = make_five_task_records(baseRecord, sampleId);

    for k = 1:numel(records)
        rec = records{k};
        jsonLine = jsonencode(rec);
        fprintf(fidAll, "%s\n", jsonLine);
        write_to_split(fidTrain, fidVal, fidTest, splitName, jsonLine);

        recordCount = recordCount + 1;
        taskCounts = update_task_counts(taskCounts, rec.task, splitName);
        splitCounts.(char(splitName)) = splitCounts.(char(splitName)) + 1;
    end

    if mod(sampleCount, 100) == 0
        fprintf("Processed %d samples, generated %d five-task records.\n", ...
            sampleCount, recordCount);
    end
end

write_task_counts_csv(taskCounts, taskCountPath);

summary = struct;
summary.source = inputPath;
summary.sample_manifest = manifestPath;
summary.benchmark_name = "sixtech_fivetask_benchmark";
summary.technology_count = 6;
summary.task_count = 5;
summary.samples_processed = sampleCount;
summary.records_generated = recordCount;
summary.expected_records_per_sample = 5;
summary.split_counts = splitCounts;
summary.tasks = ["technology_recognition", "snr_bucket", ...
    "time_occupancy", "frequency_occupancy", "domain_condition"];
summary.outputs = struct;
summary.outputs.all = outputPath;
summary.outputs.train = trainPath;
summary.outputs.val = valPath;
summary.outputs.test = testPath;
summary.outputs.task_counts = taskCountPath;

fidSummary = fopen(summaryPath, "w");
if fidSummary < 0
    error("Cannot open summary output: %s", summaryPath);
end
fprintf(fidSummary, "%s\n", jsonencode(summary));
fclose(fidSummary);

fprintf("\nSix-technology five-task benchmark created.\n");
fprintf("Samples processed: %d\n", sampleCount);
fprintf("Benchmark records generated: %d\n", recordCount);
fprintf("Output: %s\n", outputPath);
fprintf("Splits: train=%d, val=%d, test=%d\n", ...
    splitCounts.train, splitCounts.val, splitCounts.test);
fprintf("Task counts: %s\n", taskCountPath);
fprintf("Summary: %s\n", summaryPath);

function records = make_five_task_records(baseRecord, sampleId)
    imagePath = get_field_as_string(baseRecord, "image", "");
    metadata = get_record_metadata(baseRecord);
    technology = get_field_as_string(baseRecord, "answer", ...
        get_field_as_string(metadata, "technology", "unknown"));
    snrBucket = get_snr_bucket(metadata);
    timeLabel = get_time_occupancy_label(metadata);
    freqLabel = get_frequency_occupancy_label(metadata);
    domainLabel = get_domain_label(baseRecord, metadata);

    records = cell(5, 1);
    records{1} = make_benchmark_record(sampleId, imagePath, ...
        "technology_recognition", ...
        "Which wireless technology is shown in this RF spectrogram?", ...
        technology, ["5G NR", "LTE", "UMTS", "WLAN", "DVB-S2", "Bluetooth"], metadata);

    records{2} = make_benchmark_record(sampleId, imagePath, ...
        "snr_bucket", ...
        "What is the simulated SNR level bucket?", ...
        snrBucket, ["low", "medium", "high", "unknown"], metadata);

    records{3} = make_benchmark_record(sampleId, imagePath, ...
        "time_occupancy", ...
        "What is the approximate time occupancy pattern?", ...
        timeLabel, ["full", "single_burst", "double_burst", ...
        "periodic_burst", "no_gating", "unknown"], metadata);

    records{4} = make_benchmark_record(sampleId, imagePath, ...
        "frequency_occupancy", ...
        "What is the approximate frequency occupancy pattern?", ...
        freqLabel, ["wideband", "moderate_band", "narrowband", ...
        "low_shifted", "high_shifted", "two_subbands", ...
        "frequency_hopping", "full_spectrum", "unknown"], metadata);

    records{5} = make_benchmark_record(sampleId, imagePath, ...
        "domain_condition", ...
        "Which synthetic domain condition does this spectrogram belong to?", ...
        domainLabel, ["in_distribution", "shifted_impairment", ...
        "weak_profile", "no_profile", "unknown"], metadata);
end

function rec = make_benchmark_record(sampleId, imagePath, taskName, question, answer, candidateLabels, metadata)
    rec = struct;
    rec.id = sprintf("%s_%s", char(sampleId), char(taskName));
    rec.sample_id = char(sampleId);
    rec.image = char(imagePath);
    rec.task = char(taskName);
    rec.question = char(question);
    rec.answer = char(answer);
    rec.candidate_labels = candidateLabels;
    rec.metadata = metadata;
end

function bucket = get_snr_bucket(metadata)
    bucket = "unknown";
    if ~isstruct(metadata) || ~isfield(metadata, "snr_db") || isempty(metadata.snr_db)
        return;
    end

    snrDb = double(metadata.snr_db);
    if snrDb < 10
        bucket = "low";
    elseif snrDb < 20
        bucket = "medium";
    else
        bucket = "high";
    end
end

function label = get_time_occupancy_label(metadata)
    label = "unknown";
    if ~isstruct(metadata) || ~isfield(metadata, "gating_meta") || ...
            ~isstruct(metadata.gating_meta) || ~isfield(metadata.gating_meta, "gating_mode")
        return;
    end

    gatingMode = string(metadata.gating_meta.gating_mode);
    switch char(gatingMode)
        case 'full'
            label = "full";
        case 'single_burst'
            label = "single_burst";
        case 'double_burst'
            label = "double_burst";
        case 'periodic_burst'
            label = "periodic_burst";
        case 'none'
            label = "no_gating";
        otherwise
            label = "unknown";
    end
end

function label = get_frequency_occupancy_label(metadata)
    label = "unknown";
    if ~isstruct(metadata) || ~isfield(metadata, "freq_shape_meta") || ...
            ~isstruct(metadata.freq_shape_meta) || ~isfield(metadata.freq_shape_meta, "freq_mode")
        return;
    end

    freqMode = lower(char(string(metadata.freq_shape_meta.freq_mode)));
    if strcmp(freqMode, "none")
        label = "full_spectrum";
    elseif strcmp(freqMode, "bluetooth_hopping")
        label = "frequency_hopping";
    elseif contains(freqMode, "two")
        label = "two_subbands";
    elseif contains(freqMode, "low") || strcmp(freqMode, "low_shifted")
        label = "low_shifted";
    elseif contains(freqMode, "high") || strcmp(freqMode, "high_shifted")
        label = "high_shifted";
    elseif contains(freqMode, "narrow") || contains(freqMode, "dvbs2") || contains(freqMode, "bt_base")
        label = "narrowband";
    elseif contains(freqMode, "medium") || contains(freqMode, "lte") || contains(freqMode, "umts")
        label = "moderate_band";
    elseif contains(freqMode, "wide") || strcmp(freqMode, "fullband")
        label = "wideband";
    else
        label = "unknown";
    end
end

function domainLabel = get_domain_label(baseRecord, metadata)
    domainLabel = "unknown";
    if isstruct(baseRecord) && isfield(baseRecord, "domain") && ~isempty(baseRecord.domain)
        domainLabel = string(baseRecord.domain);
        return;
    end
    if isstruct(metadata) && isfield(metadata, "domain") && ~isempty(metadata.domain)
        domainLabel = string(metadata.domain);
    end
end

function metadata = get_record_metadata(record)
    if isstruct(record) && isfield(record, "metadata") && isstruct(record.metadata)
        metadata = record.metadata;
    else
        metadata = struct;
    end
end

function splitMap = read_split_manifest(path)
    splitMap = containers.Map("KeyType", "char", "ValueType", "char");
    fid = fopen(path, "r");
    if fid < 0
        error("Cannot open split manifest: %s", path);
    end
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    lineNumber = 0;
    while true
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        lineNumber = lineNumber + 1;
        line = strtrim(line);
        if isempty(line)
            continue;
        end

        try
            rec = jsondecode(line);
        catch ME
            warning("Skipping manifest line %d: %s", lineNumber, ME.message);
            continue;
        end

        sampleId = get_field_as_string(rec, "id", "");
        splitName = get_field_as_string(rec, "split", "");
        if sampleId ~= "" && splitName ~= ""
            splitMap(char(sampleId)) = char(splitName);
        end
    end
end

function splitName = get_split(splitMap, sampleId)
    if isKey(splitMap, char(sampleId))
        splitName = string(splitMap(char(sampleId)));
    else
        error("No split found for sample id: %s", sampleId);
    end
end

function sampleId = get_sample_id(record, fallbackIndex)
    sampleId = "";
    if isstruct(record) && isfield(record, "metadata") && ...
            isstruct(record.metadata) && isfield(record.metadata, "id")
        sampleId = get_field_as_string(record.metadata, "id", "");
    end
    if sampleId == "" && isstruct(record) && isfield(record, "sample_id")
        sampleId = get_field_as_string(record, "sample_id", "");
    end
    if sampleId == "" && isstruct(record) && isfield(record, "id")
        recordId = get_field_as_string(record, "id", "");
        sampleId = regexprep(recordId, "_wtr$", "");
    end
    if sampleId == ""
        sampleId = sprintf("sample_%06d", fallbackIndex);
    end
end

function write_to_split(fidTrain, fidVal, fidTest, splitName, jsonLine)
    switch char(splitName)
        case 'train'
            fprintf(fidTrain, "%s\n", jsonLine);
        case 'val'
            fprintf(fidVal, "%s\n", jsonLine);
        case 'test'
            fprintf(fidTest, "%s\n", jsonLine);
        otherwise
            error("Unknown split name: %s", splitName);
    end
end

function taskCounts = init_task_count_struct()
    taskNames = ["technology_recognition", "snr_bucket", ...
        "time_occupancy", "frequency_occupancy", "domain_condition"];
    splitNames = ["train", "val", "test", "all"];

    taskCounts = struct;
    for i = 1:numel(taskNames)
        fieldName = matlab.lang.makeValidName(taskNames(i));
        for j = 1:numel(splitNames)
            taskCounts.(fieldName).(char(splitNames(j))) = 0;
        end
    end
end

function splitCounts = init_split_count_struct()
    splitCounts = struct;
    splitCounts.train = 0;
    splitCounts.val = 0;
    splitCounts.test = 0;
end

function taskCounts = update_task_counts(taskCounts, taskName, splitName)
    fieldName = matlab.lang.makeValidName(char(taskName));
    taskCounts.(fieldName).all = taskCounts.(fieldName).all + 1;
    taskCounts.(fieldName).(char(splitName)) = taskCounts.(fieldName).(char(splitName)) + 1;
end

function write_task_counts_csv(taskCounts, outPath)
    fid = fopen(outPath, "w");
    if fid < 0
        error("Cannot open task count CSV: %s", outPath);
    end
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, "task,train,val,test,all\n");
    taskNames = fieldnames(taskCounts);
    for i = 1:numel(taskNames)
        counts = taskCounts.(taskNames{i});
        fprintf(fid, "%s,%d,%d,%d,%d\n", taskNames{i}, ...
            counts.train, counts.val, counts.test, counts.all);
    end
end

function value = get_field_as_string(s, fieldName, defaultValue)
    fieldName = char(fieldName);
    value = string(defaultValue);
    if ~isstruct(s) || ~isfield(s, fieldName)
        return;
    end
    rawValue = s.(fieldName);
    if isempty(rawValue)
        return;
    end
    if isnumeric(rawValue) || islogical(rawValue)
        if isscalar(rawValue)
            value = string(num2str(rawValue));
        else
            value = string(mat2str(rawValue));
        end
    elseif isstring(rawValue)
        value = rawValue;
    elseif ischar(rawValue)
        value = string(rawValue);
    else
        try
            value = string(rawValue);
        catch
            value = string(defaultValue);
        end
    end
end

function cleanup_files(varargin)
    for i = 1:nargin
        close_if_open(varargin{i});
    end
end

function close_if_open(fid)
    if isnumeric(fid) && fid > 0
        fclose(fid);
    end
end
