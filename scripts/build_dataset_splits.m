clc; clear; close all;

rootDir = "data_all";
splitDir = fullfile(rootDir, "splits");
metadataPath = fullfile(rootDir, "metadata_index.jsonl");
instructionPath = fullfile(rootDir, "instruction_data.jsonl");
wtrPath = fullfile(rootDir, "wtr_benchmark.jsonl");

if ~exist(metadataPath, "file")
    error("Metadata index not found: %s", metadataPath);
end
if ~exist(instructionPath, "file")
    error("Instruction JSONL not found: %s", instructionPath);
end
if ~exist(wtrPath, "file")
    error("WTR benchmark JSONL not found: %s", wtrPath);
end
if ~exist(splitDir, "dir")
    mkdir(splitDir);
end

splitSeed = 20260703;
rng(splitSeed);

[samples, metadataLines] = read_metadata_index(metadataPath);
splitMap = make_stratified_sample_split(samples, splitSeed);

manifestPath = fullfile(splitDir, "sample_splits.jsonl");
metadataCounts = write_metadata_splits(metadataLines, splitMap, splitDir, manifestPath);
instructionCounts = split_jsonl_by_sample(instructionPath, splitMap, splitDir, "instruction");
wtrCounts = split_jsonl_by_sample(wtrPath, splitMap, splitDir, "wtr");

summary = struct;
summary.split_seed = splitSeed;
summary.split_policy = "stratified_by_technology_sample_level";
summary.sample_counts = metadataCounts;
summary.instruction_counts = instructionCounts;
summary.wtr_counts = wtrCounts;
summary.outputs = struct;
summary.outputs.split_dir = splitDir;
summary.outputs.sample_manifest = manifestPath;
summary.outputs.instruction_train = fullfile(splitDir, "instruction_train.jsonl");
summary.outputs.instruction_val = fullfile(splitDir, "instruction_val.jsonl");
summary.outputs.instruction_test = fullfile(splitDir, "instruction_test.jsonl");
summary.outputs.wtr_train = fullfile(splitDir, "wtr_train.jsonl");
summary.outputs.wtr_val = fullfile(splitDir, "wtr_val.jsonl");
summary.outputs.wtr_test = fullfile(splitDir, "wtr_test.jsonl");

summaryPath = fullfile(splitDir, "split_summary.json");
fidSummary = fopen(summaryPath, "w");
if fidSummary < 0
    error("Cannot open split summary file: %s", summaryPath);
end
fprintf(fidSummary, "%s\n", jsonencode(summary));
fclose(fidSummary);

fprintf("\nDataset splits created.\n");
fprintf("Split seed: %d\n", splitSeed);
fprintf("Output directory: %s\n", splitDir);
fprintf("Sample counts: train=%d, val=%d, test=%d\n", ...
    metadataCounts.train, metadataCounts.val, metadataCounts.test);
fprintf("Instruction counts: train=%d, val=%d, test=%d\n", ...
    instructionCounts.train, instructionCounts.val, instructionCounts.test);
fprintf("WTR counts: train=%d, val=%d, test=%d\n", ...
    wtrCounts.train, wtrCounts.val, wtrCounts.test);
fprintf("Summary file: %s\n", summaryPath);

function [samples, metadataLines] = read_metadata_index(metadataPath)
    fid = fopen(metadataPath, "r");
    if fid < 0
        error("Cannot open metadata index: %s", metadataPath);
    end
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    samples = struct("id", {}, "technology", {}, "line", {});
    metadataLines = {};
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
            meta = jsondecode(line);
        catch ME
            warning("Skipping metadata line %d: %s", lineNumber, ME.message);
            continue;
        end

        sampleId = get_field_as_string(meta, "id", sprintf("sample_%06d", numel(samples) + 1));
        technology = get_field_as_string(meta, "technology", "unknown");

        samples(end + 1).id = sampleId; %#ok<AGROW>
        samples(end).technology = technology;
        samples(end).line = line;
        metadataLines{end + 1} = line; %#ok<AGROW>
    end
end

function splitMap = make_stratified_sample_split(samples, splitSeed)
    rng(splitSeed);
    splitMap = containers.Map("KeyType", "char", "ValueType", "char");
    sampleTechnologies = strings(1, numel(samples));
    for i = 1:numel(samples)
        sampleTechnologies(i) = string(samples(i).technology);
    end
    technologies = unique(sampleTechnologies);

    for t = 1:numel(technologies)
        tech = technologies(t);
        idx = find(sampleTechnologies == tech);
        idx = idx(randperm(numel(idx)));

        n = numel(idx);
        nTrain = floor(0.80 * n);
        nVal = floor(0.10 * n);

        trainIdx = idx(1:nTrain);
        valIdx = idx(nTrain + 1:nTrain + nVal);
        testIdx = idx(nTrain + nVal + 1:end);

        assign_split(splitMap, samples, trainIdx, "train");
        assign_split(splitMap, samples, valIdx, "val");
        assign_split(splitMap, samples, testIdx, "test");
    end
end

function assign_split(splitMap, samples, idx, splitName)
    for i = 1:numel(idx)
        splitMap(char(samples(idx(i)).id)) = char(splitName);
    end
end

function counts = write_metadata_splits(metadataLines, splitMap, splitDir, manifestPath)
    fids = open_split_files(splitDir, "metadata");
    fidManifest = fopen(manifestPath, "w");
    if fidManifest < 0
        close_split_files(fids);
        error("Cannot open sample split manifest: %s", manifestPath);
    end
    cleanupObj = onCleanup(@() cleanup_split_files(fids, fidManifest)); %#ok<NASGU>

    counts = init_counts();

    for i = 1:numel(metadataLines)
        line = metadataLines{i};
        meta = jsondecode(line);
        sampleId = get_field_as_string(meta, "id", "");
        technology = get_field_as_string(meta, "technology", "unknown");
        splitName = get_split(splitMap, sampleId);

        fprintf(fids.(splitName), "%s\n", line);
        counts.(splitName) = counts.(splitName) + 1;

        rec = struct;
        rec.id = sampleId;
        rec.technology = technology;
        rec.split = splitName;
        fprintf(fidManifest, "%s\n", jsonencode(rec));
    end
end

function counts = split_jsonl_by_sample(inputPath, splitMap, splitDir, outputPrefix)
    fidIn = fopen(inputPath, "r");
    if fidIn < 0
        error("Cannot open input JSONL: %s", inputPath);
    end
    fids = open_split_files(splitDir, outputPrefix);
    cleanupObj = onCleanup(@() cleanup_split_files(fids, fidIn)); %#ok<NASGU>

    counts = init_counts();
    lineNumber = 0;

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
            record = jsondecode(line);
        catch ME
            warning("Skipping %s line %d: %s", inputPath, lineNumber, ME.message);
            continue;
        end

        sampleId = get_sample_id(record);
        splitName = get_split(splitMap, sampleId);
        fprintf(fids.(splitName), "%s\n", line);
        counts.(splitName) = counts.(splitName) + 1;
    end
end

function sampleId = get_sample_id(record)
    sampleId = "";
    if isstruct(record) && isfield(record, "metadata") && ...
            isstruct(record.metadata) && isfield(record.metadata, "id")
        sampleId = get_field_as_string(record.metadata, "id", "");
    end
    if sampleId == "" && isstruct(record) && isfield(record, "id")
        recordId = get_field_as_string(record, "id", "");
        sampleId = regexprep(recordId, "_(technology_recognition|concise_description|snr_bucket|link_direction|time_occupancy|frequency_occupancy|channel_bandwidth_extraction|wtr)$", "");
    end
end

function splitName = get_split(splitMap, sampleId)
    if isKey(splitMap, char(sampleId))
        splitName = splitMap(char(sampleId));
    else
        error("No split assignment found for sample id: %s", sampleId);
    end
end

function fids = open_split_files(splitDir, prefix)
    splitNames = ["train", "val", "test"];
    fids = struct;
    for i = 1:numel(splitNames)
        splitName = splitNames(i);
        path = fullfile(splitDir, sprintf("%s_%s.jsonl", char(prefix), char(splitName)));
        fid = fopen(path, "w");
        if fid < 0
            close_split_files(fids);
            error("Cannot open split file: %s", path);
        end
        fids.(char(splitName)) = fid;
    end
end

function close_split_files(fids)
    names = fieldnames(fids);
    for i = 1:numel(names)
        if fids.(names{i}) > 0
            fclose(fids.(names{i}));
        end
    end
end

function cleanup_split_files(fids, extraFid)
    close_split_files(fids);
    if extraFid > 0
        fclose(extraFid);
    end
end

function counts = init_counts()
    counts = struct;
    counts.train = 0;
    counts.val = 0;
    counts.test = 0;
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
