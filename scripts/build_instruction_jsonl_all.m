clc; clear; close all;

inputPath = fullfile("data_all", "metadata_index.jsonl");
outputPath = fullfile("data_all", "instruction_data.jsonl");

if ~exist(inputPath, "file")
    error("Input metadata JSONL file not found: %s", inputPath);
end

fidIn = fopen(inputPath, "r");
if fidIn < 0
    error("Cannot open input file: %s", inputPath);
end

fidOut = fopen(outputPath, "w");
if fidOut < 0
    fclose(fidIn);
    error("Cannot open output file: %s", outputPath);
end

samplesProcessed = 0;
instructionsGenerated = 0;
taskCounts = struct;
instructionIds = {};
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
        meta = jsondecode(line);
    catch ME
        warning("Skipping line %d due to JSON parse error: %s", lineNumber, ME.message);
        continue;
    end

    samplesProcessed = samplesProcessed + 1;

    sampleId = getFieldAsString(meta, "id", sprintf("sample_%06d", samplesProcessed));
    imagePath = getFieldAsString(meta, "spectrogram_path", "");
    technology = getFieldAsString(meta, "technology", "unknown");
    standard = getFieldAsString(meta, "standard", get_nested_field(meta, "generator_meta", "standard", "unknown"));
    linkDirection = getFieldAsString(meta, "link_direction", "unknown");
    snrBucket = getSnrBucket(meta);

    answer = sprintf("It is a synthetic %s signal.", technology);
    if strcmpi(technology, '5G NR')
        answer = "It is a synthetic 5G NR signal.";
    end
    [instructionsGenerated, taskCounts, instructionIds] = emitInstruction(fidOut, ...
        sampleId, imagePath, "technology_recognition", ...
        "Which wireless technology is shown in this RF spectrogram?", ...
        answer, meta, instructionsGenerated, taskCounts, instructionIds);

    answer = sprintf("This is a synthetic %s waveform generated in MATLAB.", technology);
    if ~strcmp(standard, "unknown")
        answer = sprintf("This is a synthetic %s %s waveform generated in MATLAB.", ...
            technology, standard);
    end
    [instructionsGenerated, taskCounts, instructionIds] = emitInstruction(fidOut, ...
        sampleId, imagePath, "concise_description", ...
        "Give a concise RF description of this spectrogram.", ...
        answer, meta, instructionsGenerated, taskCounts, instructionIds);

    answer = sprintf("The simulated SNR level is %s.", char(snrBucket));
    [instructionsGenerated, taskCounts, instructionIds] = emitInstruction(fidOut, ...
        sampleId, imagePath, "snr_bucket", ...
        "Is the simulated SNR level low, medium, or high?", ...
        answer, meta, instructionsGenerated, taskCounts, instructionIds);

    if isfield(meta, 'link_direction')
        if strcmpi(technology, '5G NR')
            question = "Is this 5G NR scene downlink or uplink?";
        else
            question = "What is the link direction of this RF scene?";
        end
        answer = sprintf("The link direction is %s.", linkDirection);
        [instructionsGenerated, taskCounts, instructionIds] = emitInstruction(fidOut, ...
            sampleId, imagePath, "link_direction", question, answer, ...
            meta, instructionsGenerated, taskCounts, instructionIds);
    end

    if isfield(meta, 'gating_meta') && isfield(meta.gating_meta, 'gating_mode')
        gatingMode = getFieldAsString(meta.gating_meta, "gating_mode", "unknown");
        answer = getTimeOccupancyAnswer(gatingMode);
        [instructionsGenerated, taskCounts, instructionIds] = emitInstruction(fidOut, ...
            sampleId, imagePath, "time_occupancy", ...
            "What is the approximate time occupancy pattern of this RF scene?", ...
            answer, meta, instructionsGenerated, taskCounts, instructionIds);
    end

    if isfield(meta, 'freq_shape_meta') && isfield(meta.freq_shape_meta, 'freq_mode')
        freqMode = getFieldAsString(meta.freq_shape_meta, "freq_mode", "unknown");
        answer = getFrequencyOccupancyAnswer(freqMode);
        [instructionsGenerated, taskCounts, instructionIds] = emitInstruction(fidOut, ...
            sampleId, imagePath, "frequency_occupancy", ...
            "What is the approximate frequency occupancy pattern?", ...
            answer, meta, instructionsGenerated, taskCounts, instructionIds);
    end

    if isfield(meta, 'generator_meta') && isfield(meta.generator_meta, 'channel_bandwidth')
        channelBandwidth = getFieldAsString(meta.generator_meta, "channel_bandwidth", "unknown");
        answer = sprintf("The channel bandwidth is %s MHz.", channelBandwidth);
        [instructionsGenerated, taskCounts, instructionIds] = emitInstruction(fidOut, ...
            sampleId, imagePath, "channel_bandwidth_extraction", ...
            "What channel bandwidth was used for this waveform?", ...
            answer, meta, instructionsGenerated, taskCounts, instructionIds);
    end

    if mod(samplesProcessed, 50) == 0
        fprintf("Processed %d samples, generated %d instructions.\n", ...
            samplesProcessed, instructionsGenerated);
    end
end

fclose(fidIn);
fclose(fidOut);

fprintf("\nSamples processed: %d\n", samplesProcessed);
fprintf("Instructions generated: %d\n", instructionsGenerated);
fprintf("Output file: %s\n", outputPath);
fprintf("Task counts:\n");
printTaskCounts(taskCounts);
checkUniqueInstructionIds(instructionIds);

function value = getFieldAsString(s, fieldName, defaultValue)
    fieldName = char(fieldName);
    value = char(defaultValue);
    if ~isstruct(s) || ~isfield(s, fieldName)
        return;
    end
    rawValue = s.(fieldName);
    if isempty(rawValue)
        return;
    end
    if isnumeric(rawValue) || islogical(rawValue)
        if isscalar(rawValue)
            value = num2str(rawValue);
        else
            value = mat2str(rawValue);
        end
    elseif isstring(rawValue)
        value = char(rawValue);
    elseif ischar(rawValue)
        value = rawValue;
    else
        try
            value = char(string(rawValue));
        catch
            value = char(defaultValue);
        end
    end
end

function value = get_nested_field(s, structField, valueField, defaultValue)
    value = defaultValue;
    if isstruct(s) && isfield(s, char(structField)) && ...
            isstruct(s.(char(structField))) && isfield(s.(char(structField)), char(valueField))
        value = getFieldAsString(s.(char(structField)), valueField, defaultValue);
    end
end

function bucket = getSnrBucket(meta)
    bucket = "unknown";
    if ~isstruct(meta) || ~isfield(meta, 'snr_db') || isempty(meta.snr_db)
        return;
    end
    snrDb = double(meta.snr_db);
    if snrDb < 10
        bucket = "low";
    elseif snrDb < 20
        bucket = "medium";
    else
        bucket = "high";
    end
end

function inst = makeInstruction(sampleId, imagePath, taskName, question, answer, metadata)
    inst = struct;
    inst.id = sprintf("%s_%s", char(sampleId), char(taskName));
    inst.image = char(imagePath);
    inst.task = char(taskName);
    inst.question = char(question);
    inst.answer = char(answer);
    inst.metadata = metadata;
end

function writeInstruction(fid, inst)
    fprintf(fid, "%s\n", jsonencode(inst));
end

function taskCounts = updateCount(taskCounts, taskName)
    fieldName = matlab.lang.makeValidName(char(taskName));
    if ~isfield(taskCounts, fieldName)
        taskCounts.(fieldName) = 0;
    end
    taskCounts.(fieldName) = taskCounts.(fieldName) + 1;
end

function [instructionsGenerated, taskCounts, instructionIds] = emitInstruction(fid, sampleId, imagePath, ...
        taskName, question, answer, metadata, instructionsGenerated, taskCounts, instructionIds)
    inst = makeInstruction(sampleId, imagePath, taskName, question, answer, metadata);
    writeInstruction(fid, inst);
    taskCounts = updateCount(taskCounts, taskName);
    instructionIds{end + 1} = inst.id; %#ok<AGROW>
    instructionsGenerated = instructionsGenerated + 1;
end

function answer = getTimeOccupancyAnswer(gatingMode)
    switch char(gatingMode)
        case 'full'
            answer = "The signal is active across most of the observation window.";
        case 'single_burst'
            answer = "The signal appears as a single burst in time.";
        case 'double_burst'
            answer = "The signal appears as two separated bursts in time.";
        case 'periodic_burst'
            answer = "The signal appears as multiple periodic bursts in time.";
        otherwise
            answer = "The time occupancy pattern is unknown.";
    end
end

function answer = getFrequencyOccupancyAnswer(freqMode)
    switch char(freqMode)
        case {'fullband', 'wide_center', 'wide_low', 'wide_high', ...
                'nr_wide_center', 'wlan_wide_center', 'wlan_wide_low', 'wlan_wide_high'}
            answer = "The signal occupies a wide frequency band.";
        case {'narrowband', 'narrow_center', 'narrow_low', 'narrow_high', ...
                'dvbs2_narrow_center', 'dvbs2_narrow_low', 'dvbs2_narrow_high', ...
                'very_narrow_low', 'very_narrow_high', 'bt_base_narrow'}
            answer = "The signal occupies a relatively narrow frequency band.";
        case {'medium_center', 'lte_medium_center', 'umts_mid_center'}
            answer = "The signal occupies a moderate frequency band near the center.";
        case {'low_shifted', 'medium_low', 'lte_medium_low', 'umts_mid_low'}
            answer = "The signal energy is shifted toward lower frequencies.";
        case {'high_shifted', 'medium_high', 'lte_medium_high', 'umts_mid_high'}
            answer = "The signal energy is shifted toward higher frequencies.";
        case {'two_subbands', 'two_tiny_subbands'}
            answer = "The signal occupies two separated frequency subbands.";
        case 'bluetooth_hopping'
            answer = "The signal appears as narrowband energy hopping across multiple frequency positions.";
        otherwise
            answer = "The frequency occupancy pattern is unknown.";
    end
end

function printTaskCounts(taskCounts)
    taskNames = fieldnames(taskCounts);
    for i = 1:numel(taskNames)
        fprintf("  %s: %d\n", taskNames{i}, taskCounts.(taskNames{i}));
    end
end

function checkUniqueInstructionIds(instructionIds)
    if isempty(instructionIds)
        warning("No instruction IDs were generated.");
        return;
    end
    instructionIds = cellfun(@char, instructionIds, 'UniformOutput', false);
    uniqueIds = unique(instructionIds);
    if numel(uniqueIds) ~= numel(instructionIds)
        warning("Duplicate instruction IDs found.");
    else
        fprintf("All instruction IDs are unique.\n");
    end
end
