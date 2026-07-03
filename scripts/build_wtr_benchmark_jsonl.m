clc; clear; close all;

inputPath = fullfile("data_all", "metadata_index.jsonl");
outputPath = fullfile("data_all", "wtr_benchmark.jsonl");

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

sampleCount = 0;
taskCount = 0;
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

    sampleCount = sampleCount + 1;
    sampleId = getFieldAsString(meta, "id", sprintf("sample_%06d", sampleCount));
    technology = getFieldAsString(meta, "technology", "unknown");
    imagePath = getFieldAsString(meta, "spectrogram_path", "");

    record = struct;
    record.id = sprintf("%s_wtr", sampleId);
    record.image = imagePath;
    record.task = "wireless_technology_recognition";
    record.question = "Which wireless technology is shown in this RF spectrogram?";
    record.answer = technology;
    record.candidate_labels = ["5G NR", "LTE", "UMTS", "WLAN", "DVB-S2", "Bluetooth"];
    record.metadata = meta;

    fprintf(fidOut, "%s\n", jsonencode(record));
    taskCount = taskCount + 1;

    if mod(sampleCount, 100) == 0
        fprintf("Processed %d samples for WTR benchmark.\n", sampleCount);
    end
end

fclose(fidIn);
fclose(fidOut);

fprintf("\nWTR benchmark samples processed: %d\n", sampleCount);
fprintf("WTR tasks generated: %d\n", taskCount);
fprintf("Output file: %s\n", outputPath);

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
