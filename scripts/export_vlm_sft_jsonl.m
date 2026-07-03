clc; clear; close all;

rootDir = "data_all";
splitDir = fullfile(rootDir, "splits");
exportDir = fullfile(rootDir, "vlm_sft");

if ~exist(splitDir, "dir")
    error("Split directory not found: %s", splitDir);
end
if ~exist(exportDir, "dir")
    mkdir(exportDir);
end

splitNames = ["train", "val", "test"];
totalRecords = 0;

for i = 1:numel(splitNames)
    splitName = splitNames(i);
    inputPath = fullfile(splitDir, "instruction_" + splitName + ".jsonl");
    outputPath = fullfile(exportDir, "llava_" + splitName + ".jsonl");

    count = export_split(inputPath, outputPath);
    totalRecords = totalRecords + count;
    fprintf("Exported %s: %d records -> %s\n", splitName, count, outputPath);
end

manifest = struct;
manifest.format = "llava_style_jsonl";
manifest.description = "Synthetic RF spectrogram visual instruction data exported from MATLAB metadata.";
manifest.image_root = ".";
manifest.train_file = fullfile(exportDir, "llava_train.jsonl");
manifest.val_file = fullfile(exportDir, "llava_val.jsonl");
manifest.test_file = fullfile(exportDir, "llava_test.jsonl");
manifest.total_records = totalRecords;
manifest.note = "Each record contains image, conversations, task, source_id, and metadata. Data are synthetic, not real OTA captures.";

manifestPath = fullfile(exportDir, "manifest.json");
fidManifest = fopen(manifestPath, "w");
if fidManifest < 0
    error("Cannot open VLM export manifest: %s", manifestPath);
end
fprintf(fidManifest, "%s\n", jsonencode(manifest));
fclose(fidManifest);

fprintf("\nVLM SFT export complete.\n");
fprintf("Output directory: %s\n", exportDir);
fprintf("Manifest: %s\n", manifestPath);

function count = export_split(inputPath, outputPath)
    if ~exist(inputPath, "file")
        error("Input instruction split not found: %s", inputPath);
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

    cleanupObj = onCleanup(@() cleanup_files(fidIn, fidOut)); %#ok<NASGU>
    count = 0;
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
            inst = jsondecode(line);
        catch ME
            warning("Skipping %s line %d: %s", inputPath, lineNumber, ME.message);
            continue;
        end

        record = make_llava_record(inst);
        fprintf(fidOut, "%s\n", jsonencode(record));
        count = count + 1;
    end
end

function record = make_llava_record(inst)
    question = get_field_as_string(inst, "question", "");
    answer = get_field_as_string(inst, "answer", "");

    humanTurn = struct;
    humanTurn.from = "human";
    humanTurn.value = sprintf("<image>\n%s", question);

    gptTurn = struct;
    gptTurn.from = "gpt";
    gptTurn.value = answer;

    record = struct;
    record.id = get_field_as_string(inst, "id", "");
    record.image = normalize_path(get_field_as_string(inst, "image", ""));
    record.conversations = [humanTurn; gptTurn];
    record.task = get_field_as_string(inst, "task", "unknown");
    record.source_id = get_source_id(inst);

    if isfield(inst, "metadata")
        record.metadata = inst.metadata;
    else
        record.metadata = struct;
    end
end

function sourceId = get_source_id(inst)
    sourceId = "";
    if isstruct(inst) && isfield(inst, "metadata") && ...
            isstruct(inst.metadata) && isfield(inst.metadata, "id")
        sourceId = get_field_as_string(inst.metadata, "id", "");
    end
end

function value = normalize_path(value)
    value = string(value);
    value = replace(value, "\", "/");
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

function cleanup_files(fidIn, fidOut)
    if fidIn > 0
        fclose(fidIn);
    end
    if fidOut > 0
        fclose(fidOut);
    end
end

