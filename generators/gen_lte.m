function [waveform, fs, meta] = gen_lte(seed)
%GEN_LTE Generate a conservative randomized LTE downlink waveform.
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

        if exist('lteRMCDL', 'file') == 2 && exist('lteRMCDLTool', 'file') == 2
            rcChoices = {'R.3', 'R.6', 'R.8'};
            rc = rcChoices{randi(numel(rcChoices))};
            cfg = lteRMCDL(rc);

            if isfield(cfg, 'TotSubframes')
                cfg.TotSubframes = 10;
            end
            if isfield(cfg, 'NCellID')
                cfg.NCellID = randi([0 503]);
            end

            numBits = sum(cfg.PDSCH.TrBlkSizes);
            trData = randi([0 1], numBits, 1);
            [waveform, ~, cfgOut] = lteRMCDLTool(cfg, trData);
            waveform = waveform(:);

            if isfield(cfgOut, 'SamplingRate')
                fs = cfgOut.SamplingRate;
            else
                fs = 15.36e6;
            end

            meta = struct;
            meta.technology = "LTE";
            meta.standard = "LTE";
            meta.seed = seed;
            meta.generator = "lteRMCDLTool";
            meta.sample_rate = fs;
            meta.reference_channel = string(rc);
            meta.num_subframes = get_struct_field(cfgOut, 'TotSubframes', 10);
            meta.ncellid = get_struct_field(cfgOut, 'NCellID', "unknown");
            meta.num_bits = numBits;
        else
            [waveform, fs, meta] = fallback_qpsk(seed, "LTE", "LTE", ...
                "fallback_qpsk_lte", 15.36e6);
        end
    catch ME
        try
            [waveform, fs, meta] = fallback_qpsk(seed, "LTE", "LTE", ...
                "fallback_qpsk_lte_after_lteRMCDLTool_error", 15.36e6);
            meta.fallback_reason = string(ME.message);
        catch fallbackME
            error("gen_lte:GenerationFailed", ...
                "Failed to generate LTE waveform with gen_lte for seed %s: %s", ...
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
