clc; clear; close all;

outputRoot = "data_robust";
splitDir = fullfile(outputRoot, "splits");

inputSets = struct([]);
inputSets(1).path = fullfile("data_all", "wtr_benchmark.jsonl");
inputSets(1).domain = "in_distribution";
inputSets(2).path = fullfile("data_hard", "shifted_impairment", "wtr_benchmark.jsonl");
inputSets(2).domain = "shifted_impairment";
inputSets(3).path = fullfile("data_hard", "weak_profile", "wtr_benchmark.jsonl");
inputSets(3).domain = "weak_profile";
inputSets(4).path = fullfile("data_hard", "no_profile", "wtr_benchmark.jsonl");
inputSets(4).domain = "no_profile";

splitSeed = 20260704;
trainRatio = 0.70;
valRatio = 0.15;

ensure_dir(outputRoot);
ensure_dir(splitDir);

for i = 1:numel(inputSets)
    if ~exist(inputSets(i).path, "file")
        error("Input WTR JSONL not found: %s", inputSets(i).path);
    end
end

records = {};
sampleIds = strings(0, 1);
technologies = strings(0, 1);
domains = strings(0, 1);

for i = 1:numel(inputSets)
    [newRecords, newSampleIds, newTechnologies, newDomains] = ...
        read_wtr_records(inputSets(i).path, inputSets(i).domain);

    records = [records; newRecords]; %#ok<AGROW>
    sampleIds = [sampleIds; newSampleIds]; %#ok<AGROW>
    technologies = [technologies; newTechnologies]; %#ok<AGROW>
    domains = [domains; newDomains]; %#ok<AGROW>

    fprintf("Loaded %d records from %s as domain=%s\n", ...
        numel(newRecords), inputSets(i).path, inputSets(i).domain);
end

if isempty(records)
    error("No WTR records loaded.");
end

assert_unique_sample_ids(sampleIds);

rng(splitSeed);
splits = make_domain_technology_splits(domains, technologies, trainRatio, valRatio);

allPath = fullfile(outputRoot, "wtr_benchmark.jsonl");
manifestPath = fullfile(splitDir, "sample_manifest.jsonl");
write_all_records(records, allPath);
splitCounts = write_split_records(records, sampleIds, technologies, domains, splits, splitDir, manifestPath);

groupSummaryPath = fullfile(splitDir, "group_summary.csv");
write_group_summary(domains, technologies, splits, groupSummaryPath);

summary = struct;
summary.split_seed = splitSeed;
summary.split_policy = "stratified_by_domain_and_technology_sample_level";
summary.train_ratio = trainRatio;
summary.val_ratio = valRatio;
summary.test_ratio = 1 - trainRatio - valRatio;
summary.total_records = numel(records);
summary.split_counts = splitCounts;
summary.outputs = struct;
summary.outputs.root = outputRoot;
summary.outputs.wtr_benchmark = allPath;
summary.outputs.split_dir = splitDir;
summary.outputs.wtr_train = fullfile(splitDir, "wtr_train.jsonl");
summary.outputs.wtr_val = fullfile(splitDir, "wtr_val.jsonl");
summary.outputs.wtr_test = fullfile(splitDir, "wtr_test.jsonl");
summary.outputs.sample_manifest = manifestPath;
summary.outputs.group_summary = groupSummaryPath;

summaryPath = fullfile(splitDir, "split_summary.json");
fidSummary = fopen(summaryPath, "w");
if fidSummary < 0
    error("Cannot open split summary file: %s", summaryPath);
end
fprintf(fidSummary, "%s\n", jsonencode(summary));
fclose(fidSummary);

fprintf("\nRobust WTR splits created.\n");
fprintf("Output root: %s\n", outputRoot);
fprintf("Total WTR records: %d\n", numel(records));
fprintf("Split seed: %d\n", splitSeed);
fprintf("WTR counts: train=%d, val=%d, test=%d\n", ...
    splitCounts.train, splitCounts.val, splitCounts.test);
fprintf("Group summary: %s\n", groupSummaryPath);
fprintf("Summary file: %s\n", summaryPath);

function [records, sampleIds, technologies, domains] = read_wtr_records(path, domainName)
    fid = fopen(path, "r");
    if fid < 0
        error("Cannot open WTR JSONL: %s", path);
    end
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    records = {};
    sampleIds = strings(0, 1);
    technologies = strings(0, 1);
    domains = strings(0, 1);
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
            record = jsondecode(line);
        catch ME
            warning("Skipping %s line %d: %s", path, lineNumber, ME.message);
            continue;
        end

        if ~isfield(record, "metadata") || ~isstruct(record.metadata)
            record.metadata = struct;
        end

        record.domain = string(domainName);
        record.metadata.domain = string(domainName);

        sampleId = get_sample_id(record);
        technology = get_field_as_string(record, "answer", ...
            get_field_as_string(record.metadata, "technology", "unknown"));

        records{end + 1, 1} = record; %#ok<AGROW>
        sampleIds(end + 1, 1) = sampleId; %#ok<AGROW>
        technologies(end + 1, 1) = technology; %#ok<AGROW>
        domains(end + 1, 1) = string(domainName); %#ok<AGROW>
    end
end

function splits = make_domain_technology_splits(domains, technologies, trainRatio, valRatio)
    splits = strings(numel(domains), 1);
    groupKeys = domains + "||" + technologies;
    uniqueGroups = unique(groupKeys);

    for g = 1:numel(uniqueGroups)
        idx = find(groupKeys == uniqueGroups(g));
        idx = idx(randperm(numel(idx)));

        n = numel(idx);
        nTrain = floor(trainRatio * n);
        nVal = floor(valRatio * n);

        trainIdx = idx(1:nTrain);
        valIdx = idx(nTrain + 1:nTrain + nVal);
        testIdx = idx(nTrain + nVal + 1:end);

        splits(trainIdx) = "train";
        splits(valIdx) = "val";
        splits(testIdx) = "test";
    end

    if any(splits == "")
        error("Internal split assignment error: some records were not assigned.");
    end
end

function write_all_records(records, outPath)
    fid = fopen(outPath, "w");
    if fid < 0
        error("Cannot open output WTR JSONL: %s", outPath);
    end
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    for i = 1:numel(records)
        fprintf(fid, "%s\n", jsonencode(records{i}));
    end
end

function splitCounts = write_split_records(records, sampleIds, technologies, domains, splits, splitDir, manifestPath)
    fids = open_split_files(splitDir);
    fidManifest = fopen(manifestPath, "w");
    if fidManifest < 0
        close_split_files(fids);
        error("Cannot open sample manifest file: %s", manifestPath);
    end
    cleanupObj = onCleanup(@() cleanup_split_files(fids, fidManifest)); %#ok<NASGU>

    splitCounts = init_counts();

    for i = 1:numel(records)
        splitName = char(splits(i));
        fprintf(fids.(splitName), "%s\n", jsonencode(records{i}));
        splitCounts.(splitName) = splitCounts.(splitName) + 1;

        rec = struct;
        rec.id = sampleIds(i);
        rec.technology = technologies(i);
        rec.domain = domains(i);
        rec.split = string(splitName);
        rec.image = get_field_as_string(records{i}, "image", "");
        fprintf(fidManifest, "%s\n", jsonencode(rec));
    end
end

function write_group_summary(domains, technologies, splits, outPath)
    uniqueDomains = unique(domains);
    uniqueTechs = unique(technologies);

    rows = {};
    for d = 1:numel(uniqueDomains)
        for t = 1:numel(uniqueTechs)
            mask = domains == uniqueDomains(d) & technologies == uniqueTechs(t);
            if ~any(mask)
                continue;
            end
            row = struct;
            row.domain = uniqueDomains(d);
            row.technology = uniqueTechs(t);
            row.train = sum(mask & splits == "train");
            row.val = sum(mask & splits == "val");
            row.test = sum(mask & splits == "test");
            row.total = sum(mask);
            rows{end + 1, 1} = row; %#ok<AGROW>
        end
    end

    fid = fopen(outPath, "w");
    if fid < 0
        error("Cannot open group summary CSV: %s", outPath);
    end
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, "domain,technology,train,val,test,total\n");
    for i = 1:numel(rows)
        row = rows{i};
        fprintf(fid, "%s,%s,%d,%d,%d,%d\n", ...
            row.domain, row.technology, row.train, row.val, row.test, row.total);
    end
end

function assert_unique_sample_ids(sampleIds)
    [uniqueIds, ~, groupIdx] = unique(sampleIds);
    counts = accumarray(groupIdx, 1);
    duplicateIds = uniqueIds(counts > 1);
    if ~isempty(duplicateIds)
        warning("Found %d duplicate sample ids. First duplicate: %s", ...
            numel(duplicateIds), duplicateIds(1));
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
        sampleId = regexprep(recordId, "_wtr$", "");
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

function fids = open_split_files(splitDir)
    splitNames = ["train", "val", "test"];
    fids = struct;
    for i = 1:numel(splitNames)
        splitName = splitNames(i);
        path = fullfile(splitDir, sprintf("wtr_%s.jsonl", char(splitName)));
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

function ensure_dir(folder)
    if ~exist(folder, "dir")
        mkdir(folder);
    end
end
