clc; clear; close all;

addpath("generators");
addpath("utils");

if ~exist("data_wlan", "dir")
    mkdir("data_wlan");
end

if ~exist("data_wlan/iq", "dir")
    mkdir("data_wlan/iq");
end

if ~exist("data_wlan/spectrograms", "dir")
    mkdir("data_wlan/spectrograms");
end

if ~exist("data_wlan/metadata", "dir")
    mkdir("data_wlan/metadata");
end

cleanup_previous_generated_outputs();

numSamples = 100;
baseSeed = 3026;

metadataIndexPath = "data_wlan/metadata_index.jsonl";
fid = fopen(metadataIndexPath, "w");
if fid < 0
    error("Cannot open metadata index file: %s", metadataIndexPath);
end

successCount = 0;

for i = 1:numSamples
    fidMeta = -1;
    try
        sampleId = sprintf("wlan_%06d", i);
        sampleIdStr = string(sampleId);
        seed = baseSeed + i;

        [waveform, fs, genMeta] = gen_wlan(seed);
        [waveform, gatingMeta] = apply_random_time_gating(waveform, seed + 100000);
        [waveform, freqMeta] = apply_random_frequency_shaping(waveform, seed + 200000);

        snr_db = 5 + (25 - 5) * rand;
        freq_offset_hz = (-fs * 1e-4) + (2 * fs * 1e-4) * rand;

        waveform = apply_freq_offset(waveform, fs, freq_offset_hz);
        waveform = add_awgn_custom(waveform, snr_db);

        waveformPath = fullfile("data_wlan", "iq", sampleIdStr + ".mat");
        spectrogramPath = fullfile("data_wlan", "spectrograms", sampleIdStr + ".png");
        metadataPath = fullfile("data_wlan", "metadata", sampleIdStr + ".json");

        save(waveformPath, "waveform", "fs", "snr_db", ...
            "freq_offset_hz", "seed", "genMeta", "gatingMeta", "freqMeta");

        make_spectrogram_image(waveform, fs, spectrogramPath);

        metadata = struct;
        metadata.id = sampleId;
        metadata.technology = genMeta.technology;
        metadata.waveform_path = waveformPath;
        metadata.spectrogram_path = spectrogramPath;
        metadata.fs = fs;
        metadata.snr_db = snr_db;
        metadata.freq_offset_hz = freq_offset_hz;
        metadata.seed = seed;
        metadata.generator_meta = genMeta;
        metadata.gating_meta = gatingMeta;
        metadata.freq_shape_meta = freqMeta;

        metadataJson = jsonencode(metadata);

        fidMeta = fopen(metadataPath, "w");
        if fidMeta < 0
            error("Cannot open metadata file: %s", metadataPath);
        end
        fprintf(fidMeta, "%s\n", metadataJson);
        fclose(fidMeta);
        fidMeta = -1;

        fprintf(fid, "%s\n", metadataJson);
        successCount = successCount + 1;

        if mod(i, 10) == 0
            fprintf("Generated %d/%d WLAN samples, success = %d\n", ...
                i, numSamples, successCount);
        end
    catch ME
        if fidMeta > 0
            fclose(fidMeta);
        end
        warning("Failed to generate WLAN sample %d: %s", i, ME.message);
    end
end

fclose(fid);

fprintf("Finished WLAN dataset generation.\n");
fprintf("Successful samples: %d/%d\n", successCount, numSamples);
fprintf("Output path: %s\n", fullfile(pwd, "data_wlan"));

function cleanup_previous_generated_outputs()
    delete_matching_files(fullfile("data_wlan", "iq", "wlan_*.mat"));
    delete_matching_files(fullfile("data_wlan", "spectrograms", "wlan_*.png"));
    delete_matching_files(fullfile("data_wlan", "metadata", "wlan_*.json"));
end

function delete_matching_files(pattern)
    files = dir(pattern);
    for k = 1:numel(files)
        filePath = fullfile(files(k).folder, files(k).name);
        try
            delete(filePath);
        catch ME
            warning("Could not delete old generated file %s: %s", filePath, ME.message);
        end
    end
end
