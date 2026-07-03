clc; clear; close all;

inputPath = fullfile("data_robust", "wtr_benchmark.jsonl");
outputRoot = "data_domain_holdout";
holdoutDomains = ["shifted_impairment", "weak_profile", "no_profile"];
splitSeed = 20260705;
trainRatio = 0.85;

if ~exist(inputPath, "file")
    error("Robust WTR benchmark not found: %s. Run build_robust_wtr_splits first.", inputPath);
end

ensure_dir(outputRoot);

[records, sampleIds, technologies, domains] = read_wtr_records(inputPath);
fprintf("Loaded %d robust WTR records from %s\n", numel(records), inputPath);

for h = 1:numel(holdoutDomains)
    holdoutDomain = holdoutDomains(h);
    domainRoot = fullfile(outputRoot, holdoutDomain);
    splitDir = fullfile(domainRoot, "splits");
    ensure_dir(domainRoot);
    ensure_dir(splitDir);

    rng(splitSeed + h);
    splits = make_holdout_splits(domains, technologies, holdoutDomain, trainRatio);

    splitCounts = write_split_records(records, sampleIds, technologies, domains, splits, splitDir);
    groupSummaryPath = fullfile(splitDir, "group_summary.csv");
    write_group_summary(domains, technologies, splits, groupSummaryPath);

    summary = struct;
    summary.split_seed = splitSeed + h;
    summary.split_policy = "domain_held_out_by_domain_and_technology";
    summary.holdout_domain = holdoutDomain;
    summary.train_ratio_within_non_holdout_groups = trainRatio;
    summary.total_records = numel(records);
    summary.split_counts = splitCounts;
    summary.outputs = struct;
    summary.outputs.split_dir = splitDir;
    summary.outputs.wtr_train = fullfile(splitDir, "wtr_train.jsonl");
    summary.outputs.wtr_val = fullfile(splitDir, "wtr_val.jsonl");
    summary.outputs.wtr_test = fullfile(splitDir, "wtr_test.jsonl");
    summary.outputs.group_summary = groupSummaryPath;

    summaryPath = fullfile(splitDir, "split_summary.json");
    fidSummary = fopen(summaryPath, "w");
    if fidSummary < 0
        error("Cannot open split summary file: %s", summaryPath);
    end
    fprintf(fidSummary, "%s\n", jsonencode(summary));
    fclose(fidSummary);

    fprintf("\nCreated holdout split: %s\n", holdoutDomain);
    fprintf("Counts: train=%d, val=%d, test=%d\n", ...
        splitCounts.train, splitCounts.val, splitCounts.test);
    fprintf("Group summary: %s\n", groupSummaryPath);
end

fprintf("\nDomain-held-out WTR splits created under: %s\n", outputRoot);

function [records, sampleIds, technologies, domains] = read_wtr_records(path)
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

        sampleId = get_sample_id(record);
        technology = get_field_as_string(record, "answer", ...
            get_field_as_string(record.metadata, "technology", "unknown"));
        domainName = get_record_domain(record);

        records{end + 1, 1} = record; %#ok<AGROW>
        sampleIds(end + 1, 1) = sampleId; %#ok<AGROW>
        technologies(end + 1, 1) = technology; %#ok<AGROW>
        domains(end + 1, 1) = domainName; %#ok<AGROW>
    end
end

function splits = make_holdout_splits(domains, technologies, holdoutDomain, trainRatio)
    splits = strings(numel(domains), 1);
    testIdx = find(domains == holdoutDomain);
    if isempty(testIdx)
        error("Holdout domain has no samples: %s", holdoutDomain);
    end
    splits(testIdx) = "test";

    trainValIdx = find(domains ~= holdoutDomain);
    groupKeys = domains + "||" + technologies;
    uniqueGroups = unique(groupKeys(trainValIdx));

    for g = 1:numel(uniqueGroups)
        idx = find(groupKeys == uniqueGroups(g));
        idx = idx(domains(idx) ~= holdoutDomain);
        idx = idx(randperm(numel(idx)));

        n = numel(idx);
        nTrain = floor(trainRatio * n);
        trainIdx = idx(1:nTrain);
        valIdx = idx(nTrain + 1:end);

        splits(trainIdx) = "train";
        splits(valIdx) = "val";
    end

    if any(splits == "")
        error("Internal split assignment error: some records were not assigned.");
    end
end

function splitCounts = write_split_records(records, sampleIds, technologies, domains, splits, splitDir)
    fids = open_split_files(splitDir);
    manifestPath = fullfile(splitDir, "sample_manifest.jsonl");
    fidManifest = fopen(manifestPath, "w");
    if fidManifest < 0
        close_split_files(fids);
        error("Cannot open sample manifest: %s", manifestPath);
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

    fid = fopen(outPath, "w");
    if fid < 0
        error("Cannot open group summary CSV: %s", outPath);
    end
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, "domain,technology,train,val,test,total\n");
    for d = 1:numel(uniqueDomains)
        for t = 1:numel(uniqueTechs)
            mask = domains == uniqueDomains(d) & technologies == uniqueTechs(t);
            if ~any(mask)
                continue;
            end
            fprintf(fid, "%s,%s,%d,%d,%d,%d\n", ...
                uniqueDomains(d), uniqueTechs(t), ...
                sum(mask & splits == "train"), ...
                sum(mask & splits == "val"), ...
                sum(mask & splits == "test"), ...
                sum(mask));
        end
    end
end

function domainName = get_record_domain(record)
    domainName = "unknown";
    if isstruct(record) && isfield(record, "domain") && ~isempty(record.domain)
        domainName = string(record.domain);
        return;
    end
    if isstruct(record) && isfield(record, "metadata") && isstruct(record.metadata) && ...
            isfield(record.metadata, "domain") && ~isempty(record.metadata.domain)
        domainName = string(record.metadata.domain);
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
