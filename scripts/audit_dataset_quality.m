clc; clear; close all;

rootDir = "data_all";
outputDir = "outputs";
auditPath = fullfile(outputDir, "dataset_quality_audit.txt");

if ~exist(outputDir, "dir")
    mkdir(outputDir);
end

fidReport = fopen(auditPath, "w");
if fidReport < 0
    error("Cannot open audit report: %s", auditPath);
end
cleanupObj = onCleanup(@() fclose(fidReport)); %#ok<NASGU>

allowedTechs = ["5G NR", "LTE", "UMTS", "WLAN", "DVB-S2", "Bluetooth"];
splitNames = ["train", "val", "test"];

log_line(fidReport, "Mini RF-GPT Dataset Quality Audit");
log_line(fidReport, "Root directory: %s", rootDir);
log_line(fidReport, "Audit time: %s", string(datetime("now")));
log_line(fidReport, "");

metadataPath = fullfile(rootDir, "metadata_index.jsonl");
instructionPath = fullfile(rootDir, "instruction_data.jsonl");
wtrPath = fullfile(rootDir, "wtr_benchmark.jsonl");
splitDir = fullfile(rootDir, "splits");

requiredPaths = [metadataPath, instructionPath, wtrPath, splitDir];
missingRequired = 0;
for i = 1:numel(requiredPaths)
    if ~exist(requiredPaths(i), "file") && ~exist(requiredPaths(i), "dir")
        log_line(fidReport, "MISSING required path: %s", requiredPaths(i));
        missingRequired = missingRequired + 1;
    end
end
if missingRequired > 0
    error("Dataset audit stopped because %d required paths are missing.", missingRequired);
end

[metadataRecords, metadataDecodeErrors] = read_jsonl(metadataPath);
[instructionRecords, instructionDecodeErrors] = read_jsonl(instructionPath);
[wtrRecords, wtrDecodeErrors] = read_jsonl(wtrPath);

log_line(fidReport, "JSONL Integrity");
log_line(fidReport, "  Metadata records: %d, decode errors: %d", numel(metadataRecords), metadataDecodeErrors);
log_line(fidReport, "  Instruction records: %d, decode errors: %d", numel(instructionRecords), instructionDecodeErrors);
log_line(fidReport, "  WTR records: %d, decode errors: %d", numel(wtrRecords), wtrDecodeErrors);
log_line(fidReport, "");

sampleInfo = extract_sample_info(metadataRecords);
sampleIds = sampleInfo.ids;
sampleTechs = sampleInfo.techs;

duplicateSampleCount = numel(sampleIds) - numel(unique(sampleIds));
unknownTechCount = sum(~ismember(sampleTechs, allowedTechs));

log_line(fidReport, "Sample Inventory");
log_line(fidReport, "  Samples: %d", numel(sampleIds));
log_line(fidReport, "  Duplicate sample IDs: %d", duplicateSampleCount);
log_line(fidReport, "  Unknown technology labels: %d", unknownTechCount);
write_label_counts(fidReport, "  Technology counts", sampleTechs, allowedTechs);
log_line(fidReport, "");

sampleFileStats = audit_sample_files_and_images(sampleInfo);
log_line(fidReport, "File and Image Quality");
log_line(fidReport, "  Missing waveform MAT files: %d", sampleFileStats.missingWaveform);
log_line(fidReport, "  Missing spectrogram PNG files: %d", sampleFileStats.missingImage);
log_line(fidReport, "  Unreadable images: %d", sampleFileStats.unreadableImage);
log_line(fidReport, "  Non-512x512 images: %d", sampleFileStats.non512);
log_line(fidReport, "  Near-constant images: %d", sampleFileStats.nearConstant);
log_line(fidReport, "  Mean normalized image std: %.6f", mean(sampleFileStats.imageStd, "omitnan"));
log_line(fidReport, "  Min normalized image std: %.6f", min(sampleFileStats.imageStd, [], "omitnan"));
log_line(fidReport, "  Max normalized image std: %.6f", max(sampleFileStats.imageStd, [], "omitnan"));
log_line(fidReport, "");

metadataStats = audit_metadata_values(metadataRecords, allowedTechs);
log_line(fidReport, "Metadata Sanity");
log_line(fidReport, "  Nonpositive or missing fs: %d", metadataStats.badFs);
log_line(fidReport, "  SNR outside [5,25] dB: %d", metadataStats.badSnr);
log_line(fidReport, "  Missing/invalid frequency offset: %d", metadataStats.badFreqOffset);
log_line(fidReport, "  Missing gating metadata: %d", metadataStats.missingGating);
log_line(fidReport, "  Missing frequency shaping metadata: %d", metadataStats.missingFreqShape);
log_line(fidReport, "  Missing technology profile metadata: %d", metadataStats.missingProfile);
log_line(fidReport, "  DVB-S2 fallback samples: %d", metadataStats.dvbs2Fallback);
log_line(fidReport, "");

instructionStats = audit_instruction_records(instructionRecords, sampleIds);
log_line(fidReport, "Instruction Data");
log_line(fidReport, "  Instructions: %d", numel(instructionRecords));
log_line(fidReport, "  Duplicate instruction IDs: %d", instructionStats.duplicateIds);
log_line(fidReport, "  Instructions with unknown sample ID: %d", instructionStats.unknownSample);
log_line(fidReport, "  Forbidden exact-value tasks: %d", instructionStats.forbiddenTasks);
write_map_counts(fidReport, "  Task counts", instructionStats.taskCounts);
log_line(fidReport, "");

wtrStats = audit_wtr_records(wtrRecords, sampleIds, allowedTechs);
log_line(fidReport, "WTR Benchmark");
log_line(fidReport, "  WTR records: %d", numel(wtrRecords));
log_line(fidReport, "  Duplicate WTR IDs: %d", wtrStats.duplicateIds);
log_line(fidReport, "  WTR records with unknown sample ID: %d", wtrStats.unknownSample);
log_line(fidReport, "  WTR records with invalid answer label: %d", wtrStats.invalidAnswer);
log_line(fidReport, "");

[splitMap, splitStats] = audit_splits(splitDir, sampleIds, sampleTechs, allowedTechs, splitNames);
log_line(fidReport, "Sample-Level Splits");
log_line(fidReport, "  Split records: %d", splitStats.records);
log_line(fidReport, "  Samples missing split assignment: %d", splitStats.missingAssignments);
log_line(fidReport, "  Split records with unknown samples: %d", splitStats.unknownSamples);
log_line(fidReport, "  Split leakage issues: %d", splitStats.leakageIssues);
for i = 1:numel(splitNames)
    splitName = splitNames(i);
    log_line(fidReport, "  %s samples: %d", splitName, splitStats.counts.(char(splitName)));
    write_label_counts(fidReport, "    by technology", splitStats.techs.(char(splitName)), allowedTechs);
end
log_line(fidReport, "");

splitJsonlStats = audit_split_jsonl_files(splitDir, splitMap, splitNames);
log_line(fidReport, "Split JSONL Consistency");
log_line(fidReport, "  Metadata split violations: %d", splitJsonlStats.metadataViolations);
log_line(fidReport, "  Instruction split violations: %d", splitJsonlStats.instructionViolations);
log_line(fidReport, "  WTR split violations: %d", splitJsonlStats.wtrViolations);
log_line(fidReport, "");

baselineStats = run_visual_wtr_baseline(sampleInfo, splitMap, allowedTechs);
log_line(fidReport, "Simple Visual WTR Baseline");
log_line(fidReport, "  Nearest-centroid test accuracy: %.2f%% (%d/%d)", ...
    100 * baselineStats.centroidAccuracy, baselineStats.centroidCorrect, baselineStats.testCount);
log_line(fidReport, "  1-NN cosine test accuracy: %.2f%% (%d/%d)", ...
    100 * baselineStats.nnAccuracy, baselineStats.nnCorrect, baselineStats.testCount);
for i = 1:numel(allowedTechs)
    tech = allowedTechs(i);
    log_line(fidReport, "  %s centroid accuracy: %.2f%% (%d/%d)", ...
        tech, 100 * baselineStats.perClassAccuracy(i), ...
        baselineStats.perClassCorrect(i), baselineStats.perClassTotal(i));
end
log_line(fidReport, "  Nearest-centroid confusion matrix rows=true, columns=predicted:");
write_confusion_matrix(fidReport, allowedTechs, baselineStats.centroidConfusion);
log_line(fidReport, "  1-NN confusion matrix rows=true, columns=predicted:");
write_confusion_matrix(fidReport, allowedTechs, baselineStats.nnConfusion);
log_line(fidReport, "");

criticalIssues = metadataDecodeErrors + instructionDecodeErrors + wtrDecodeErrors + ...
    duplicateSampleCount + unknownTechCount + sampleFileStats.missingWaveform + ...
    sampleFileStats.missingImage + sampleFileStats.unreadableImage + sampleFileStats.non512 + ...
    sampleFileStats.nearConstant + metadataStats.badFs + metadataStats.badSnr + ...
    metadataStats.badFreqOffset + metadataStats.missingGating + metadataStats.missingFreqShape + ...
    metadataStats.missingProfile + instructionStats.duplicateIds + instructionStats.unknownSample + ...
    instructionStats.forbiddenTasks + wtrStats.duplicateIds + wtrStats.unknownSample + ...
    wtrStats.invalidAnswer + splitStats.missingAssignments + splitStats.unknownSamples + ...
    splitStats.leakageIssues + splitJsonlStats.metadataViolations + ...
    splitJsonlStats.instructionViolations + splitJsonlStats.wtrViolations;

log_line(fidReport, "Quality Decision");
if criticalIssues == 0 && baselineStats.centroidAccuracy >= 0.80 && baselineStats.nnAccuracy >= 0.80
    log_line(fidReport, "  PASS: Dataset is internally consistent and visually learnable for the mini WTR task.");
elseif criticalIssues == 0
    log_line(fidReport, "  CAUTION: Dataset is internally consistent, but visual separability baseline is below 80%%.");
else
    log_line(fidReport, "  FAIL: Dataset has %d critical integrity issues that should be fixed before training.", criticalIssues);
end
log_line(fidReport, "");
log_line(fidReport, "Known Limitations");
log_line(fidReport, "  - Synthetic data only; it should not be presented as real over-the-air RF captures.");
log_line(fidReport, "  - Technology profiles intentionally strengthen visual class separation for mini-WTR.");
log_line(fidReport, "  - Current scenes are mostly single-signal; dense multi-signal RF scene reasoning is not covered yet.");
log_line(fidReport, "  - DVB-S2 may use a fallback waveform if the official toolbox data is unavailable.");
log_line(fidReport, "  - Instruction answers are template-generated, not LLM-diversified dense captions.");
log_line(fidReport, "");
log_line(fidReport, "Audit report saved to: %s", auditPath);

function [records, decodeErrors] = read_jsonl(path)
    fid = fopen(path, "r");
    if fid < 0
        error("Cannot open JSONL file: %s", path);
    end
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    records = {};
    decodeErrors = 0;
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
            records{end + 1} = jsondecode(line); %#ok<AGROW>
        catch ME
            decodeErrors = decodeErrors + 1;
            warning("Failed to parse %s line %d: %s", path, lineNumber, ME.message);
        end
    end
end

function sampleInfo = extract_sample_info(metadataRecords)
    n = numel(metadataRecords);
    sampleInfo.records = metadataRecords;
    sampleInfo.ids = strings(n, 1);
    sampleInfo.techs = strings(n, 1);
    sampleInfo.images = strings(n, 1);
    sampleInfo.waveforms = strings(n, 1);
    sampleInfo.fs = nan(n, 1);
    sampleInfo.snr = nan(n, 1);

    for i = 1:n
        meta = metadataRecords{i};
        sampleInfo.ids(i) = get_field_as_string(meta, "id", "");
        sampleInfo.techs(i) = get_field_as_string(meta, "technology", "unknown");
        sampleInfo.images(i) = get_field_as_string(meta, "spectrogram_path", "");
        sampleInfo.waveforms(i) = get_field_as_string(meta, "waveform_path", "");
        sampleInfo.fs(i) = get_field_as_double(meta, "fs", nan);
        sampleInfo.snr(i) = get_field_as_double(meta, "snr_db", nan);
    end
end

function stats = audit_sample_files_and_images(sampleInfo)
    n = numel(sampleInfo.ids);
    stats.missingWaveform = 0;
    stats.missingImage = 0;
    stats.unreadableImage = 0;
    stats.non512 = 0;
    stats.nearConstant = 0;
    stats.imageStd = nan(n, 1);

    for i = 1:n
        if ~exist(sampleInfo.waveforms(i), "file")
            stats.missingWaveform = stats.missingWaveform + 1;
        end
        if ~exist(sampleInfo.images(i), "file")
            stats.missingImage = stats.missingImage + 1;
            continue;
        end

        try
            img = imread(sampleInfo.images(i));
        catch
            stats.unreadableImage = stats.unreadableImage + 1;
            continue;
        end

        if size(img, 1) ~= 512 || size(img, 2) ~= 512
            stats.non512 = stats.non512 + 1;
        end

        gray = normalize_gray(img);
        stats.imageStd(i) = std(gray(:));
        if stats.imageStd(i) < 1e-4
            stats.nearConstant = stats.nearConstant + 1;
        end
    end
end

function stats = audit_metadata_values(metadataRecords, allowedTechs)
    stats.badFs = 0;
    stats.badSnr = 0;
    stats.badFreqOffset = 0;
    stats.missingGating = 0;
    stats.missingFreqShape = 0;
    stats.missingProfile = 0;
    stats.dvbs2Fallback = 0;

    for i = 1:numel(metadataRecords)
        meta = metadataRecords{i};
        fs = get_field_as_double(meta, "fs", nan);
        snrDb = get_field_as_double(meta, "snr_db", nan);
        freqOffset = get_field_as_double(meta, "freq_offset_hz", nan);
        technology = get_field_as_string(meta, "technology", "unknown");

        if ~isfinite(fs) || fs <= 0
            stats.badFs = stats.badFs + 1;
        end
        if ~isfinite(snrDb) || snrDb < 5 || snrDb > 25
            stats.badSnr = stats.badSnr + 1;
        end
        if ~isfinite(freqOffset)
            stats.badFreqOffset = stats.badFreqOffset + 1;
        end
        if ~isfield(meta, "gating_meta")
            stats.missingGating = stats.missingGating + 1;
        end
        if ~isfield(meta, "freq_shape_meta")
            stats.missingFreqShape = stats.missingFreqShape + 1;
        end
        if ~isfield(meta, "profile_meta")
            stats.missingProfile = stats.missingProfile + 1;
        end
        if technology == "DVB-S2" && isfield(meta, "generator_meta") && ...
                isstruct(meta.generator_meta) && isfield(meta.generator_meta, "fallback_reason")
            stats.dvbs2Fallback = stats.dvbs2Fallback + 1;
        end
        if ~ismember(technology, allowedTechs)
            warning("Unknown technology label in metadata: %s", technology);
        end
    end
end

function stats = audit_instruction_records(records, sampleIds)
    stats.duplicateIds = 0;
    stats.unknownSample = 0;
    stats.forbiddenTasks = 0;
    stats.taskCounts = containers.Map("KeyType", "char", "ValueType", "double");

    ids = strings(numel(records), 1);
    forbiddenTasks = ["exact_snr", "snr_extraction", "frequency_offset_extraction", "exact_frequency_offset"];
    sampleSet = containers.Map(cellstr(sampleIds), num2cell(true(size(sampleIds))));

    for i = 1:numel(records)
        rec = records{i};
        ids(i) = get_field_as_string(rec, "id", "");
        task = get_field_as_string(rec, "task", "unknown");
        taskKey = char(task);
        if ~isKey(stats.taskCounts, taskKey)
            stats.taskCounts(taskKey) = 0;
        end
        stats.taskCounts(taskKey) = stats.taskCounts(taskKey) + 1;

        if ismember(task, forbiddenTasks)
            stats.forbiddenTasks = stats.forbiddenTasks + 1;
        end

        sampleId = get_sample_id_from_record(rec);
        if ~isKey(sampleSet, char(sampleId))
            stats.unknownSample = stats.unknownSample + 1;
        end
    end

    stats.duplicateIds = numel(ids) - numel(unique(ids));
end

function stats = audit_wtr_records(records, sampleIds, allowedTechs)
    stats.duplicateIds = 0;
    stats.unknownSample = 0;
    stats.invalidAnswer = 0;

    ids = strings(numel(records), 1);
    sampleSet = containers.Map(cellstr(sampleIds), num2cell(true(size(sampleIds))));

    for i = 1:numel(records)
        rec = records{i};
        ids(i) = get_field_as_string(rec, "id", "");
        sampleId = get_sample_id_from_record(rec);
        answer = get_field_as_string(rec, "answer", "unknown");

        if ~isKey(sampleSet, char(sampleId))
            stats.unknownSample = stats.unknownSample + 1;
        end
        if ~ismember(answer, allowedTechs)
            stats.invalidAnswer = stats.invalidAnswer + 1;
        end
    end

    stats.duplicateIds = numel(ids) - numel(unique(ids));
end

function [splitMap, stats] = audit_splits(splitDir, sampleIds, sampleTechs, ~, splitNames)
    manifestPath = fullfile(splitDir, "sample_splits.jsonl");
    [records, decodeErrors] = read_jsonl(manifestPath);
    if decodeErrors > 0
        warning("sample_splits.jsonl has %d decode errors.", decodeErrors);
    end

    splitMap = containers.Map("KeyType", "char", "ValueType", "char");
    sampleSet = containers.Map(cellstr(sampleIds), num2cell(true(size(sampleIds))));

    stats.records = numel(records);
    stats.missingAssignments = 0;
    stats.unknownSamples = 0;
    stats.leakageIssues = 0;
    stats.counts = init_split_count_struct(splitNames);
    stats.techs = init_split_tech_struct(splitNames);

    for i = 1:numel(records)
        rec = records{i};
        sampleId = get_field_as_string(rec, "id", "");
        splitName = get_field_as_string(rec, "split", "");

        if ~isKey(sampleSet, char(sampleId))
            stats.unknownSamples = stats.unknownSamples + 1;
            continue;
        end
        if isKey(splitMap, char(sampleId))
            stats.leakageIssues = stats.leakageIssues + 1;
            continue;
        end
        splitMap(char(sampleId)) = char(splitName);
    end

    for i = 1:numel(sampleIds)
        sampleId = sampleIds(i);
        tech = sampleTechs(i);
        if ~isKey(splitMap, char(sampleId))
            stats.missingAssignments = stats.missingAssignments + 1;
            continue;
        end
        splitName = splitMap(char(sampleId));
        if ~isfield(stats.counts, splitName)
            stats.leakageIssues = stats.leakageIssues + 1;
            continue;
        end
        stats.counts.(splitName) = stats.counts.(splitName) + 1;
        stats.techs.(splitName)(end + 1, 1) = tech;
    end

end

function stats = audit_split_jsonl_files(splitDir, splitMap, splitNames)
    stats.metadataViolations = 0;
    stats.instructionViolations = 0;
    stats.wtrViolations = 0;

    for i = 1:numel(splitNames)
        splitName = splitNames(i);
        stats.metadataViolations = stats.metadataViolations + ...
            count_split_violations(fullfile(splitDir, "metadata_" + splitName + ".jsonl"), splitMap, splitName);
        stats.instructionViolations = stats.instructionViolations + ...
            count_split_violations(fullfile(splitDir, "instruction_" + splitName + ".jsonl"), splitMap, splitName);
        stats.wtrViolations = stats.wtrViolations + ...
            count_split_violations(fullfile(splitDir, "wtr_" + splitName + ".jsonl"), splitMap, splitName);
    end
end

function violations = count_split_violations(path, splitMap, expectedSplit)
    violations = 0;
    if ~exist(path, "file")
        violations = violations + 1;
        return;
    end

    [records, decodeErrors] = read_jsonl(path);
    violations = violations + decodeErrors;

    for i = 1:numel(records)
        sampleId = get_sample_id_from_record(records{i});
        if ~isKey(splitMap, char(sampleId))
            violations = violations + 1;
        elseif string(splitMap(char(sampleId))) ~= expectedSplit
            violations = violations + 1;
        end
    end
end

function stats = run_visual_wtr_baseline(sampleInfo, splitMap, allowedTechs)
    trainIdx = false(numel(sampleInfo.ids), 1);
    testIdx = false(numel(sampleInfo.ids), 1);
    for i = 1:numel(sampleInfo.ids)
        sampleId = sampleInfo.ids(i);
        if ~isKey(splitMap, char(sampleId))
            continue;
        end
        splitName = string(splitMap(char(sampleId)));
        trainIdx(i) = splitName == "train";
        testIdx(i) = splitName == "test";
    end

    trainImages = sampleInfo.images(trainIdx);
    testImages = sampleInfo.images(testIdx);
    trainLabels = sampleInfo.techs(trainIdx);
    testLabels = sampleInfo.techs(testIdx);

    targetSize = [128 128];
    trainVectors = load_image_vectors(trainImages, targetSize);
    testVectors = load_image_vectors(testImages, targetSize);

    trainVectors = normalize_rows(trainVectors);
    testVectors = normalize_rows(testVectors);

    centroids = zeros(numel(allowedTechs), size(trainVectors, 2), "single");
    for i = 1:numel(allowedTechs)
        idx = trainLabels == allowedTechs(i);
        centroids(i, :) = mean(trainVectors(idx, :), 1);
    end
    centroids = normalize_rows(centroids);

    centroidSimilarity = testVectors * centroids.';
    [~, centroidIdx] = max(centroidSimilarity, [], 2);
    centroidPred = reshape(allowedTechs(centroidIdx), [], 1);
    centroidCorrectMask = centroidPred == testLabels;

    nnSimilarity = testVectors * trainVectors.';
    [~, nnIdx] = max(nnSimilarity, [], 2);
    nnPred = trainLabels(nnIdx);
    nnCorrectMask = nnPred == testLabels;

    stats.testCount = numel(testLabels);
    stats.centroidCorrect = sum(centroidCorrectMask);
    stats.centroidAccuracy = stats.centroidCorrect / max(stats.testCount, 1);
    stats.nnCorrect = sum(nnCorrectMask);
    stats.nnAccuracy = stats.nnCorrect / max(stats.testCount, 1);
    stats.perClassCorrect = zeros(numel(allowedTechs), 1);
    stats.perClassTotal = zeros(numel(allowedTechs), 1);
    stats.perClassAccuracy = zeros(numel(allowedTechs), 1);
    stats.centroidConfusion = zeros(numel(allowedTechs));
    stats.nnConfusion = zeros(numel(allowedTechs));

    for i = 1:numel(allowedTechs)
        idx = testLabels == allowedTechs(i);
        stats.perClassTotal(i) = sum(idx);
        stats.perClassCorrect(i) = sum(centroidCorrectMask(idx));
        stats.perClassAccuracy(i) = stats.perClassCorrect(i) / max(stats.perClassTotal(i), 1);
        for j = 1:numel(allowedTechs)
            stats.centroidConfusion(i, j) = sum(idx & centroidPred == allowedTechs(j));
            stats.nnConfusion(i, j) = sum(idx & nnPred == allowedTechs(j));
        end
    end
end

function vectors = load_image_vectors(imagePaths, targetSize)
    vectors = zeros(numel(imagePaths), prod(targetSize), "single");
    for i = 1:numel(imagePaths)
        img = imread(imagePaths(i));
        gray = normalize_gray(img);
        if size(gray, 1) ~= targetSize(1) || size(gray, 2) ~= targetSize(2)
            gray = imresize(gray, targetSize);
        end
        vectors(i, :) = single(gray(:).');
    end
end

function normalizedVectors = normalize_rows(vectors)
    rowNorms = sqrt(sum(vectors.^2, 2));
    normalizedVectors = vectors ./ max(rowNorms, eps("single"));
end

function gray = normalize_gray(img)
    if ndims(img) == 3
        gray = double(img(:, :, 1)) * 0.2989 + ...
            double(img(:, :, 2)) * 0.5870 + ...
            double(img(:, :, 3)) * 0.1140;
    else
        gray = double(img);
    end
    minVal = min(gray(:));
    maxVal = max(gray(:));
    if maxVal > minVal
        gray = (gray - minVal) / (maxVal - minVal);
    else
        gray = zeros(size(gray));
    end
end

function sampleId = get_sample_id_from_record(record)
    sampleId = "";
    if isstruct(record) && isfield(record, "metadata") && ...
            isstruct(record.metadata) && isfield(record.metadata, "id")
        sampleId = get_field_as_string(record.metadata, "id", "");
        return;
    end
    if isstruct(record) && isfield(record, "id")
        recordId = get_field_as_string(record, "id", "");
        sampleId = regexprep(recordId, "_(technology_recognition|concise_description|snr_bucket|link_direction|time_occupancy|frequency_occupancy|channel_bandwidth_extraction|wtr)$", "");
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

function value = get_field_as_double(s, fieldName, defaultValue)
    value = defaultValue;
    fieldName = char(fieldName);
    if ~isstruct(s) || ~isfield(s, fieldName)
        return;
    end
    rawValue = s.(fieldName);
    if isempty(rawValue)
        return;
    end
    if isnumeric(rawValue) || islogical(rawValue)
        value = double(rawValue(1));
    elseif ischar(rawValue) || isstring(rawValue)
        value = str2double(rawValue);
    end
end

function counts = init_split_count_struct(splitNames)
    counts = struct;
    for i = 1:numel(splitNames)
        counts.(char(splitNames(i))) = 0;
    end
end

function techs = init_split_tech_struct(splitNames)
    techs = struct;
    for i = 1:numel(splitNames)
        techs.(char(splitNames(i))) = strings(0, 1);
    end
end

function write_label_counts(fid, titleText, labels, allowedLabels)
    log_line(fid, "%s:", titleText);
    for i = 1:numel(allowedLabels)
        label = allowedLabels(i);
        log_line(fid, "    %s: %d", label, sum(labels == label));
    end
end

function write_map_counts(fid, titleText, mapValue)
    log_line(fid, "%s:", titleText);
    keysValue = sort(string(keys(mapValue)));
    for i = 1:numel(keysValue)
        key = char(keysValue(i));
        log_line(fid, "    %s: %d", key, mapValue(key));
    end
end

function write_confusion_matrix(fid, labels, matrixValue)
    fprintf("%14s", "");
    fprintf(fid, "%14s", "");
    for j = 1:numel(labels)
        fprintf("%14s", char(labels(j)));
        fprintf(fid, "%14s", char(labels(j)));
    end
    fprintf("\n");
    fprintf(fid, "\n");

    for i = 1:numel(labels)
        fprintf("%14s", char(labels(i)));
        fprintf(fid, "%14s", char(labels(i)));
        for j = 1:numel(labels)
            fprintf("%14d", matrixValue(i, j));
            fprintf(fid, "%14d", matrixValue(i, j));
        end
        fprintf("\n");
        fprintf(fid, "\n");
    end
end

function log_line(fid, fmt, varargin)
    if nargin == 2
        fprintf("%s\n", fmt);
        fprintf(fid, "%s\n", fmt);
    else
        fprintf(fmt + "\n", varargin{:});
        fprintf(fid, fmt + "\n", varargin{:});
    end
end
