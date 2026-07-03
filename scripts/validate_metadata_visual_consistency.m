clc; clear; close all;

metadataIndexPath = fullfile("data", "metadata_index.jsonl");
outputDir = "outputs";
outputCsvPath = fullfile(outputDir, "metadata_visual_consistency.csv");

if ~exist(metadataIndexPath, "file")
    error("Metadata index file not found: %s", metadataIndexPath);
end

if ~exist(outputDir, "dir")
    mkdir(outputDir);
end

fid = fopen(metadataIndexPath, "r");
if fid < 0
    error("Cannot open metadata index file: %s", metadataIndexPath);
end

rows = struct( ...
    "id", {}, ...
    "gating_mode", {}, ...
    "active_time_fraction", {}, ...
    "burst_count", {}, ...
    "metadata_active_fraction", {}, ...
    "freq_mode", {}, ...
    "active_freq_fraction", {}, ...
    "frequency_centroid", {}, ...
    "metadata_active_band_fraction", {}, ...
    "two_peak_detected", {}, ...
    "spectrogram_path", {});

lineIdx = 0;

while true
    line = fgetl(fid);
    if ~ischar(line)
        break;
    end

    line = strtrim(line);
    if isempty(line)
        continue;
    end

    lineIdx = lineIdx + 1;
    metadata = jsondecode(line);

    if isfield(metadata, 'spectrogram_path')
        spectrogramPath = string(metadata.spectrogram_path);
    else
        warning("Line %d has no spectrogram_path; skipped.", lineIdx);
        continue;
    end

    if ~exist(spectrogramPath, "file")
        warning("Spectrogram not found for %s: %s", string(metadata.id), spectrogramPath);
        continue;
    end

    img = read_normalized_gray_image(spectrogramPath);

    timeEnergyProfile = mean(img, 1);
    freqEnergyProfile = mean(img, 2);

    [activeTimeFraction, burstCount] = estimate_time_activity(timeEnergyProfile);

    freqThreshold = mean(freqEnergyProfile) + 0.3 * std(freqEnergyProfile);
    activeFreqMask = freqEnergyProfile > freqThreshold;
    activeFreqFraction = mean(activeFreqMask);

    normalizedFreq = linspace(-0.5, 0.5, numel(freqEnergyProfile)).';
    freqWeights = freqEnergyProfile - min(freqEnergyProfile);
    frequencyCentroid = sum(normalizedFreq .* freqWeights) / (sum(freqWeights) + eps);
    twoPeakDetected = detect_two_freq_peaks(freqEnergyProfile);

    gatingMode = "missing";
    metadataActiveFraction = NaN;
    if isfield(metadata, 'gating_meta')
        gatingMeta = metadata.gating_meta;
        if isfield(gatingMeta, 'gating_mode')
            gatingMode = string(gatingMeta.gating_mode);
        end
        if isfield(gatingMeta, 'active_fraction')
            metadataActiveFraction = double(gatingMeta.active_fraction);
        end
    end

    freqMode = "missing";
    metadataActiveBandFraction = NaN;
    if isfield(metadata, 'freq_shape_meta')
        freqMeta = metadata.freq_shape_meta;
        if isfield(freqMeta, 'freq_mode')
            freqMode = string(freqMeta.freq_mode);
        end
        if isfield(freqMeta, 'active_band_fraction')
            metadataActiveBandFraction = double(freqMeta.active_band_fraction);
        end
    end

    row.id = string(metadata.id);
    row.gating_mode = gatingMode;
    row.active_time_fraction = activeTimeFraction;
    row.burst_count = burstCount;
    row.metadata_active_fraction = metadataActiveFraction;
    row.freq_mode = freqMode;
    row.active_freq_fraction = activeFreqFraction;
    row.frequency_centroid = frequencyCentroid;
    row.metadata_active_band_fraction = metadataActiveBandFraction;
    row.two_peak_detected = twoPeakDetected;
    row.spectrogram_path = spectrogramPath;

    rows(end + 1) = row; %#ok<SAGROW>
end

fclose(fid);

if isempty(rows)
    error("No valid metadata/spectrogram rows were loaded.");
end

resultsTable = struct2table(rows);
writetable(resultsTable, outputCsvPath);

fprintf("Validated %d samples.\n", height(resultsTable));
fprintf("Saved consistency table to: %s\n", outputCsvPath);

fprintf("\nGating mode summary:\n");
gatingSummary = summarize_by_group(resultsTable, "gating_mode", ...
    ["active_time_fraction", "burst_count"]);
disp(gatingSummary);

fprintf("\nFrequency mode summary:\n");
freqSummary = summarize_by_group(resultsTable, "freq_mode", ...
    ["active_freq_fraction", "frequency_centroid"]);
disp(freqSummary);

diagnose_consistency(gatingSummary, freqSummary);

function img = read_normalized_gray_image(imgPath)
    img = imread(imgPath);

    if ndims(img) == 3
        img = double(img(:, :, 1)) * 0.2989 + ...
            double(img(:, :, 2)) * 0.5870 + ...
            double(img(:, :, 3)) * 0.1140;
    else
        img = double(img);
    end

    imgMin = min(img(:));
    imgMax = max(img(:));
    if imgMax > imgMin
        img = (img - imgMin) / (imgMax - imgMin);
    else
        img = zeros(size(img));
    end
end

function segmentCount = count_true_segments(mask)
    mask = logical(mask(:));
    paddedMask = [false; mask; false];
    segmentStarts = diff(paddedMask) == 1;
    segmentCount = sum(segmentStarts);
end

function [activeTimeFraction, burstCount] = estimate_time_activity(timeEnergyProfile)
    prof = double(timeEnergyProfile(:).');
    numTimeBins = numel(prof);

    windowSize = max(5, round(0.04 * numTimeBins));
    prof = movmean(prof, windowSize);

    prof = prof - min(prof);
    prof = prof / (max(prof) + eps);

    activeMask = prof > 0.35;

    minSegmentLength = max(1, round(0.025 * numTimeBins));
    gapTolerance = max(1, round(0.015 * numTimeBins));

    activeMask = remove_short_true_segments(activeMask, minSegmentLength);
    activeMask = merge_short_false_gaps(activeMask, gapTolerance);
    activeMask = remove_short_true_segments(activeMask, minSegmentLength);

    activeTimeFraction = mean(activeMask);
    burstCount = count_true_segments(activeMask);
end

function mask = remove_short_true_segments(mask, minSegmentLength)
    mask = logical(mask(:).');
    segments = get_true_segments(mask);

    for i = 1:size(segments, 1)
        if segments(i, 2) - segments(i, 1) + 1 < minSegmentLength
            mask(segments(i, 1):segments(i, 2)) = false;
        end
    end
end

function mask = merge_short_false_gaps(mask, gapTolerance)
    mask = logical(mask(:).');
    segments = get_true_segments(mask);

    if size(segments, 1) < 2
        return;
    end

    for i = 1:size(segments, 1)-1
        gapStart = segments(i, 2) + 1;
        gapEnd = segments(i + 1, 1) - 1;
        gapLength = gapEnd - gapStart + 1;

        if gapLength > 0 && gapLength < gapTolerance
            mask(gapStart:gapEnd) = true;
        end
    end
end

function segments = get_true_segments(mask)
    mask = logical(mask(:));
    paddedMask = [false; mask; false];
    starts = find(diff(paddedMask) == 1);
    ends = find(diff(paddedMask) == -1) - 1;
    segments = [starts, ends];
end

function hasTwoPeaks = detect_two_freq_peaks(profile)
    profile = double(profile(:));
    if numel(profile) < 3
        hasTwoPeaks = false;
        return;
    end

    profile = profile - min(profile);
    if max(profile) > 0
        profile = profile / max(profile);
    end

    threshold = mean(profile) + 0.3 * std(profile);
    isPeak = profile(2:end-1) > profile(1:end-2) & ...
        profile(2:end-1) >= profile(3:end) & ...
        profile(2:end-1) > threshold;
    peakIdx = find(isPeak) + 1;

    if numel(peakIdx) < 2
        hasTwoPeaks = false;
        return;
    end

    minPeakDistance = max(4, round(0.08 * numel(profile)));
    keptPeaks = peakIdx(1);
    for i = 2:numel(peakIdx)
        if all(abs(peakIdx(i) - keptPeaks) >= minPeakDistance)
            keptPeaks(end + 1) = peakIdx(i); %#ok<AGROW>
        end
    end

    hasTwoPeaks = numel(keptPeaks) >= 2;
end

function summaryTable = summarize_by_group(inputTable, groupVar, valueVars)
    groupVarName = char(groupVar);
    valueVarNames = cellstr(valueVars);
    groups = unique(inputTable.(groupVarName), 'stable');
    groupOut = strings(numel(groups), 1);
    countOut = zeros(numel(groups), 1);
    meanValues = zeros(numel(groups), numel(valueVarNames));

    for i = 1:numel(groups)
        groupMask = inputTable.(groupVarName) == groups(i);
        groupOut(i) = groups(i);
        countOut(i) = sum(groupMask);

        for j = 1:numel(valueVarNames)
            values = inputTable.(valueVarNames{j});
            groupValues = values(groupMask);
            groupValues = groupValues(~isnan(groupValues));
            if isempty(groupValues)
                meanValues(i, j) = NaN;
            else
                meanValues(i, j) = mean(groupValues);
            end
        end
    end

    summaryTable = table(groupOut, countOut, ...
        'VariableNames', {groupVarName, 'count'});

    for j = 1:numel(valueVarNames)
        summaryTable.(['mean_' valueVarNames{j}]) = meanValues(:, j);
    end
end

function diagnose_consistency(gatingSummary, freqSummary)
    fullTime = get_summary_value(gatingSummary, "gating_mode", ...
        "full", "mean_active_time_fraction");
    fullBurstCount = get_summary_value(gatingSummary, "gating_mode", ...
        "full", "mean_burst_count");
    singleTime = get_summary_value(gatingSummary, "gating_mode", ...
        "single_burst", "mean_active_time_fraction");

    lowCentroid = get_summary_value(freqSummary, "freq_mode", ...
        "low_shifted", "mean_frequency_centroid");
    highCentroid = get_summary_value(freqSummary, "freq_mode", ...
        "high_shifted", "mean_frequency_centroid");

    narrowFreq = get_summary_value(freqSummary, "freq_mode", ...
        "narrowband", "mean_active_freq_fraction");
    fullFreq = get_summary_value(freqSummary, "freq_mode", ...
        "fullband", "mean_active_freq_fraction");

    fprintf("\nDiagnostic conclusions:\n");

    if ~isnan(fullBurstCount) && fullBurstCount > 5
        warning("Full mode still fragmented; check spectrogram thresholding or gating implementation.");
    end

    if ~isnan(fullTime) && ~isnan(singleTime) && abs(fullTime - singleTime) < 0.08
        warning("single_burst and full have very similar active_time_fraction.");
    else
        fprintf("Time gating visual separation looks reasonable.\n");
    end

    if ~isnan(lowCentroid) && ~isnan(highCentroid) && abs(highCentroid - lowCentroid) < 0.08
        warning("low_shifted and high_shifted have very similar frequency centroids.");
    else
        fprintf("Low/high shifted frequency centroids look separated.\n");
    end

    if ~isnan(narrowFreq) && ~isnan(fullFreq) && abs(fullFreq - narrowFreq) < 0.08
        warning("narrowband and fullband have very similar active_freq_fraction.");
    else
        fprintf("Narrow/fullband active frequency fractions look separated.\n");
    end
end

function value = get_summary_value(summaryTable, groupVar, groupName, valueVar)
    value = NaN;
    groupVarName = char(groupVar);
    valueVarName = char(valueVar);
    groupMask = summaryTable.(groupVarName) == string(groupName);
    if any(groupMask) && ismember(valueVarName, summaryTable.Properties.VariableNames)
        values = summaryTable.(valueVarName);
        value = values(find(groupMask, 1));
    end
end
