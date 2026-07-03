clc; clear; close all;

addpath("generators");
addpath("utils");

rootDir = "data_hard";
numSamplesPerTech = 30;

techs = struct([]);
techs(1).name = "5G NR";     techs(1).prefix = "5g_nr";     techs(1).baseSeed = 12026; techs(1).fn = @gen_5g_nr;
techs(2).name = "LTE";       techs(2).prefix = "lte";       techs(2).baseSeed = 14026; techs(2).fn = @gen_lte;
techs(3).name = "UMTS";      techs(3).prefix = "umts";      techs(3).baseSeed = 15026; techs(3).fn = @gen_umts;
techs(4).name = "WLAN";      techs(4).prefix = "wlan";      techs(4).baseSeed = 13026; techs(4).fn = @gen_wlan;
techs(5).name = "DVB-S2";    techs(5).prefix = "dvbs2";     techs(5).baseSeed = 16026; techs(5).fn = @gen_dvbs2;
techs(6).name = "Bluetooth"; techs(6).prefix = "bluetooth"; techs(6).baseSeed = 17026; techs(6).fn = @gen_bluetooth;

domains = struct([]);
domains(1).name = "shifted_impairment"; domains(1).tag = "shifted"; domains(1).description = "technology profile kept, impairment/rendering range shifted";
domains(2).name = "weak_profile";       domains(2).tag = "weak";    domains(2).description = "generic gating and frequency shaping without technology-specific profile";
domains(3).name = "no_profile";         domains(3).tag = "raw";     domains(3).description = "raw generator waveform with impairments only";

ensure_dir(rootDir);

totalSuccess = 0;
totalAttempts = 0;

for d = 1:numel(domains)
    domainName = string(domains(d).name);
    domainRoot = fullfile(rootDir, domainName);
    iqDir = fullfile(domainRoot, "iq");
    spectrogramDir = fullfile(domainRoot, "spectrograms");
    metadataDir = fullfile(domainRoot, "metadata");

    ensure_dir(domainRoot);
    ensure_dir(iqDir);
    ensure_dir(spectrogramDir);
    ensure_dir(metadataDir);

    metadataIndexPath = fullfile(domainRoot, "metadata_index.jsonl");
    wtrPath = fullfile(domainRoot, "wtr_benchmark.jsonl");

    fidMetaIndex = fopen(metadataIndexPath, "w");
    if fidMetaIndex < 0
        error("Cannot open metadata index file: %s", metadataIndexPath);
    end

    fidWtr = fopen(wtrPath, "w");
    if fidWtr < 0
        fclose(fidMetaIndex);
        error("Cannot open WTR benchmark file: %s", wtrPath);
    end

    domainSuccess = 0;
    domainAttempts = 0;

    fprintf("\n===== Generating hard-test domain: %s =====\n", domainName);

    for t = 1:numel(techs)
        fprintf("\n--- %s / %s ---\n", domainName, techs(t).name);

        for i = 1:numSamplesPerTech
            fidSingleMeta = -1;
            domainAttempts = domainAttempts + 1;
            totalAttempts = totalAttempts + 1;

            try
                sampleId = sprintf("%s_%s_%06d", char(techs(t).prefix), char(domains(d).tag), i);
                sampleIdStr = string(sampleId);
                seed = techs(t).baseSeed + d * 100000 + i;

                [waveform, fs, genMeta] = techs(t).fn(seed);
                [waveform, gatingMeta, freqMeta, profileMeta] = ...
                    apply_hard_domain_processing(waveform, fs, genMeta.technology, seed, domainName);

                [snrDb, freqOffsetHz, dynamicRangeDb] = sample_impairments(fs, seed, domainName);
                waveform = apply_freq_offset(waveform, fs, freqOffsetHz);
                waveform = add_awgn_custom(waveform, snrDb, seed + 400000);
                waveform = waveform(:);

                waveformPath = fullfile(iqDir, sampleIdStr + ".mat");
                spectrogramPath = fullfile(spectrogramDir, sampleIdStr + ".png");
                metadataPath = fullfile(metadataDir, sampleIdStr + ".json");

                save(waveformPath, "waveform", "fs", "snrDb", ...
                    "freqOffsetHz", "dynamicRangeDb", "seed", ...
                    "genMeta", "gatingMeta", "freqMeta", "profileMeta");

                make_spectrogram_image(waveform, fs, spectrogramPath, dynamicRangeDb);

                metadata = struct;
                metadata.id = sampleId;
                metadata.technology = genMeta.technology;
                if isfield(genMeta, 'standard')
                    metadata.standard = genMeta.standard;
                end
                if isfield(genMeta, 'link_direction')
                    metadata.link_direction = genMeta.link_direction;
                end
                metadata.domain = domainName;
                metadata.domain_description = string(domains(d).description);
                metadata.waveform_path = waveformPath;
                metadata.spectrogram_path = spectrogramPath;
                metadata.fs = fs;
                metadata.snr_db = snrDb;
                metadata.freq_offset_hz = freqOffsetHz;
                metadata.dynamic_range_db = dynamicRangeDb;
                metadata.seed = seed;
                metadata.generator_meta = genMeta;
                metadata.gating_meta = gatingMeta;
                metadata.freq_shape_meta = freqMeta;
                metadata.profile_meta = profileMeta;

                metadataJson = jsonencode(metadata);

                fidSingleMeta = fopen(metadataPath, "w");
                if fidSingleMeta < 0
                    error("Cannot open metadata file: %s", metadataPath);
                end
                fprintf(fidSingleMeta, "%s\n", metadataJson);
                fclose(fidSingleMeta);
                fidSingleMeta = -1;

                fprintf(fidMetaIndex, "%s\n", metadataJson);
                fprintf(fidWtr, "%s\n", jsonencode(make_wtr_record(metadata)));

                domainSuccess = domainSuccess + 1;
                totalSuccess = totalSuccess + 1;

                if mod(i, 10) == 0
                    fprintf("Generated %s %s %d/%d, domain success = %d\n", ...
                        domainName, techs(t).name, i, numSamplesPerTech, domainSuccess);
                end
            catch ME
                if fidSingleMeta > 0
                    fclose(fidSingleMeta);
                end
                warning("Failed to generate %s %s sample %d: %s", ...
                    domainName, techs(t).name, i, ME.message);
            end
        end
    end

    fclose(fidMetaIndex);
    fclose(fidWtr);

    fprintf("\nFinished domain %s.\n", domainName);
    fprintf("Successful samples: %d/%d\n", domainSuccess, domainAttempts);
    fprintf("Metadata index: %s\n", metadataIndexPath);
    fprintf("WTR benchmark: %s\n", wtrPath);
end

fprintf("\nFinished hard-test dataset generation.\n");
fprintf("Successful samples: %d/%d\n", totalSuccess, totalAttempts);
fprintf("Output path: %s\n", fullfile(pwd, rootDir));

function [y, gatingMeta, freqMeta, profileMeta] = apply_hard_domain_processing(x, fs, technology, seed, domainName)
    switch char(domainName)
        case 'shifted_impairment'
            [y, gatingMeta, freqMeta, profileMeta] = ...
                apply_technology_visual_profile(x, fs, technology, seed + 100000);
            profileMeta.stress_test_domain = "shifted_impairment";
            profileMeta.stress_test_note = "technology profile kept; impairments and rendering range shifted";

        case 'weak_profile'
            [y, gatingMeta] = apply_random_time_gating(x, seed + 110000);
            [y, freqMeta] = apply_random_frequency_shaping(y, seed + 120000);

            profileMeta = struct;
            profileMeta.technology_profile = string(technology);
            profileMeta.fs = fs;
            profileMeta.seed = seed + 100000;
            profileMeta.profile_generator = "generic_random_gating_and_frequency_shaping";
            profileMeta.profile_name = "weak_generic_profile";
            profileMeta.stress_test_domain = "weak_profile";
            profileMeta.stress_test_note = "technology-specific visual profile removed";

        case 'no_profile'
            y = x(:);
            gatingMeta = make_no_gating_meta(numel(y));
            freqMeta = make_no_frequency_meta;

            profileMeta = struct;
            profileMeta.technology_profile = string(technology);
            profileMeta.fs = fs;
            profileMeta.seed = seed + 100000;
            profileMeta.profile_generator = "none";
            profileMeta.profile_name = "no_profile_raw_generator_waveform";
            profileMeta.stress_test_domain = "no_profile";
            profileMeta.stress_test_note = "raw generator waveform with only frequency offset and AWGN";

        otherwise
            error("Unknown hard-test domain: %s", domainName);
    end

    y = y(:);
end

function [snrDb, freqOffsetHz, dynamicRangeDb] = sample_impairments(fs, seed, domainName)
    rng(seed + 300000);

    switch char(domainName)
        case 'shifted_impairment'
            snrDb = -5 + 25 * rand;
            freqScale = 5e-4;
            dynamicRangeDb = 35 + 35 * rand;

        case 'weak_profile'
            snrDb = 0 + 25 * rand;
            freqScale = 2.5e-4;
            dynamicRangeDb = 35 + 40 * rand;

        case 'no_profile'
            snrDb = 5 + 20 * rand;
            freqScale = 1e-4;
            dynamicRangeDb = 40 + 40 * rand;

        otherwise
            snrDb = 5 + 20 * rand;
            freqScale = 1e-4;
            dynamicRangeDb = 60;
    end

    freqOffsetHz = (-fs * freqScale) + (2 * fs * freqScale) * rand;
end

function gatingMeta = make_no_gating_meta(numSamples)
    gatingMeta = struct;
    gatingMeta.gating_mode = "none";
    gatingMeta.active_fraction = 1;
    gatingMeta.burst_ranges = [1 numSamples];
    gatingMeta.leakage_gain = 0;
    gatingMeta.time_roll_samples = 0;
end

function freqMeta = make_no_frequency_meta()
    freqMeta = struct;
    freqMeta.freq_mode = "none";
    freqMeta.active_band_fraction = 1;
    freqMeta.band_ranges = [-0.5 0.5];
end

function record = make_wtr_record(metadata)
    record = struct;
    record.id = sprintf("%s_wtr", char(metadata.id));
    record.image = metadata.spectrogram_path;
    record.task = "wireless_technology_recognition";
    record.question = "Which wireless technology is shown in this RF spectrogram?";
    record.answer = metadata.technology;
    record.candidate_labels = ["5G NR", "LTE", "UMTS", "WLAN", "DVB-S2", "Bluetooth"];
    record.metadata = metadata;
end

function ensure_dir(folder)
    if ~exist(folder, "dir")
        mkdir(folder);
    end
end
