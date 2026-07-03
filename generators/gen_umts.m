function [waveform, fs, meta] = gen_umts(seed)
%GEN_UMTS Generate a conservative randomized UMTS downlink waveform.
%
% Inputs:
%   seed : random seed passed to rng(seed)
%
% Outputs:
%   waveform : generated complex IQ waveform as a column vector
%   fs       : sample rate in Hz
%   meta     : metadata struct describing the generated waveform

    try
        rng(seed);

        if exist('umtsDownlinkReferenceChannels', 'file') == 2 && ...
                exist('umtsDownlinkWaveformGenerator', 'file') == 2
            referenceChannel = 'RMC12.2kbps';
            cfg = umtsDownlinkReferenceChannels(referenceChannel);

            if isfield(cfg, 'TotFrames')
                cfg.TotFrames = 1;
            end
            if isfield(cfg, 'PrimaryScramblingCode')
                cfg.PrimaryScramblingCode = randi([0 511]);
            end

            waveform = umtsDownlinkWaveformGenerator(cfg);
            waveform = waveform(:);

            oversamplingRatio = get_struct_field(cfg, 'OversamplingRatio', 4);
            fs = 3.84e6 * oversamplingRatio;

            meta = struct;
            meta.technology = "UMTS";
            meta.standard = "WCDMA";
            meta.seed = seed;
            meta.generator = "umtsDownlinkWaveformGenerator";
            meta.sample_rate = fs;
            meta.reference_channel = string(referenceChannel);
            meta.total_frames = get_struct_field(cfg, 'TotFrames', 1);
            meta.oversampling_ratio = oversamplingRatio;
            meta.primary_scrambling_code = get_struct_field(cfg, ...
                'PrimaryScramblingCode', "unknown");
        else
            [waveform, fs, meta] = fallback_qpsk(seed, "UMTS", "WCDMA", ...
                "fallback_qpsk_umts", 15.36e6);
        end
    catch ME
        try
            [waveform, fs, meta] = fallback_qpsk(seed, "UMTS", "WCDMA", ...
                "fallback_qpsk_umts_after_umtsDownlinkWaveformGenerator_error", 15.36e6);
            meta.fallback_reason = string(ME.message);
        catch fallbackME
            error("gen_umts:GenerationFailed", ...
                "Failed to generate UMTS waveform with gen_umts for seed %s: %s", ...
                mat2str(seed), fallbackME.message);
        end
    end
end

function value = get_struct_field(s, fieldName, defaultValue)
    if isstruct(s) && isfield(s, fieldName)
        value = s.(fieldName);
    else
        value = defaultValue;
    end
end

function [waveform, fs, meta] = fallback_qpsk(seed, technology, standard, generatorName, fs)
    rng(seed);
    numSymbols = 4096;
    bits = randi([0 1], 2*numSymbols, 1);
    symbols = (2*bits(1:2:end)-1) + 1j*(2*bits(2:2:end)-1);
    waveform = symbols(:) / sqrt(2);

    meta = struct;
    meta.technology = technology;
    meta.standard = standard;
    meta.seed = seed;
    meta.generator = generatorName;
    meta.sample_rate = fs;
    meta.modulation = "QPSK";
    meta.num_symbols = numSymbols;
end
