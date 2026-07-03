clc; clear; close all;

addpath("generators");
addpath("utils");

rootDir = "data_all";
iqDir = fullfile(rootDir, "iq");
spectrogramDir = fullfile(rootDir, "spectrograms");
metadataDir = fullfile(rootDir, "metadata");

ensure_dir(rootDir);
ensure_dir(iqDir);
ensure_dir(spectrogramDir);
ensure_dir(metadataDir);

cleanup_previous_generated_outputs(iqDir, spectrogramDir, metadataDir);

numSamplesPerTech = 100;

techs = struct([]);
techs(1).name = "5G NR";    techs(1).prefix = "5g_nr";     techs(1).baseSeed = 2026; techs(1).fn = @gen_5g_nr;
techs(2).name = "LTE";      techs(2).prefix = "lte";       techs(2).baseSeed = 4026; techs(2).fn = @gen_lte;
techs(3).name = "UMTS";     techs(3).prefix = "umts";      techs(3).baseSeed = 5026; techs(3).fn = @gen_umts;
techs(4).name = "WLAN";     techs(4).prefix = "wlan";      techs(4).baseSeed = 3026; techs(4).fn = @gen_wlan;
techs(5).name = "DVB-S2";   techs(5).prefix = "dvbs2";     techs(5).baseSeed = 6026; techs(5).fn = @gen_dvbs2;
techs(6).name = "Bluetooth"; techs(6).prefix = "bluetooth"; techs(6).baseSeed = 7026; techs(6).fn = @gen_bluetooth;

metadataIndexPath = fullfile(rootDir, "metadata_index.jsonl");
fid = fopen(metadataIndexPath, "w");
if fid < 0
    error("Cannot open metadata index file: %s", metadataIndexPath);
end

successCount = 0;
attemptCount = 0;

for t = 1:numel(techs)
    fprintf("\n===== Generating %s =====\n", techs(t).name);

    for i = 1:numSamplesPerTech
        fidMeta = -1;
        attemptCount = attemptCount + 1;

        try
            sampleId = sprintf("%s_%06d", char(techs(t).prefix), i);
            sampleIdStr = string(sampleId);
            seed = techs(t).baseSeed + i;

            [waveform, fs, genMeta] = techs(t).fn(seed);
            [waveform, gatingMeta, freqMeta, profileMeta] = ...
                apply_technology_visual_profile(waveform, fs, genMeta.technology, seed + 100000);

            rng(seed + 300000);
            snr_db = 5 + (25 - 5) * rand;
            freq_offset_hz = (-fs * 1e-4) + (2 * fs * 1e-4) * rand;

            waveform = apply_freq_offset(waveform, fs, freq_offset_hz);
            waveform = add_awgn_custom(waveform, snr_db, seed + 400000);

            waveformPath = fullfile(iqDir, sampleIdStr + ".mat");
            spectrogramPath = fullfile(spectrogramDir, sampleIdStr + ".png");
            metadataPath = fullfile(metadataDir, sampleIdStr + ".json");

            save(waveformPath, "waveform", "fs", "snr_db", ...
                "freq_offset_hz", "seed", "genMeta", "gatingMeta", "freqMeta", "profileMeta");

            make_spectrogram_image(waveform, fs, spectrogramPath, 45);

            metadata = struct;
            metadata.id = sampleId;
            metadata.technology = genMeta.technology;
            if isfield(genMeta, 'standard')
                metadata.standard = genMeta.standard;
            end
            if isfield(genMeta, 'link_direction')
                metadata.link_direction = genMeta.link_direction;
            end
            metadata.waveform_path = waveformPath;
            metadata.spectrogram_path = spectrogramPath;
            metadata.fs = fs;
            metadata.snr_db = snr_db;
            metadata.freq_offset_hz = freq_offset_hz;
            metadata.seed = seed;
            metadata.generator_meta = genMeta;
            metadata.gating_meta = gatingMeta;
            metadata.freq_shape_meta = freqMeta;
            metadata.profile_meta = profileMeta;

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
                fprintf("Generated %s %d/%d, total success = %d\n", ...
                    techs(t).name, i, numSamplesPerTech, successCount);
            end
        catch ME
            if fidMeta > 0
                fclose(fidMeta);
            end
            warning("Failed to generate %s sample %d: %s", techs(t).name, i, ME.message);
        end
    end
end

fclose(fid);

fprintf("\nFinished all-technology dataset generation.\n");
fprintf("Successful samples: %d/%d\n", successCount, attemptCount);
fprintf("Output path: %s\n", fullfile(pwd, rootDir));

function ensure_dir(folder)
    if ~exist(folder, "dir")
        mkdir(folder);
    end
end

function cleanup_previous_generated_outputs(iqDir, spectrogramDir, metadataDir)
    prefixes = ["5g_nr", "lte", "umts", "wlan", "dvbs2", "bluetooth"];
    for i = 1:numel(prefixes)
        delete_matching_files(fullfile(iqDir, prefixes(i) + "_*.mat"));
        delete_matching_files(fullfile(spectrogramDir, prefixes(i) + "_*.png"));
        delete_matching_files(fullfile(metadataDir, prefixes(i) + "_*.json"));
    end
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
