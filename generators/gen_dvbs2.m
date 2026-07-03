function [waveform, fs, meta] = gen_dvbs2(seed)
%GEN_DVBS2 Generate a conservative randomized DVB-S2 waveform.
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

        samplesPerSymbol = randi([2 4]);
        symbolRate = 1e6;
        fs = symbolRate * samplesPerSymbol;
        modcodChoices = [1 2 3 4 5];
        modcod = modcodChoices(randi(numel(modcodChoices)));
        dfl = randi([1600 4000]);

        if exist('dvbs2WaveformGenerator', 'file') == 2
            try
                s2WaveGen = dvbs2WaveformGenerator( ...
                    'StreamFormat', 'GS', ...
                    'FECFrame', 'short', ...
                    'MODCOD', modcod, ...
                    'DFL', dfl, ...
                    'SamplesPerSymbol', samplesPerSymbol, ...
                    'RolloffFactor', 0.35);
                data = randi([0 1], dfl, 1);
                waveform = s2WaveGen(data);
                waveform = waveform(:);

                meta = struct;
                meta.technology = "DVB-S2";
                meta.standard = "DVB-S2";
                meta.seed = seed;
                meta.generator = "dvbs2WaveformGenerator";
                meta.sample_rate = fs;
                meta.stream_format = "GS";
                meta.fec_frame = "short";
                meta.modcod = modcod;
                meta.dfl = dfl;
                meta.samples_per_symbol = samplesPerSymbol;
                meta.rolloff_factor = 0.35;
                return;
            catch officialME
                officialError = officialME.message;
            end
        else
            officialError = "dvbs2WaveformGenerator unavailable";
        end

        [waveform, fs, meta] = fallback_dvbs2_like(seed, fs, samplesPerSymbol);
        meta.fallback_reason = string(officialError);
    catch ME
        try
            [waveform, fs, meta] = fallback_dvbs2_like(seed, 2e6, 2);
            meta.fallback_reason = string(ME.message);
        catch fallbackME
            error("gen_dvbs2:GenerationFailed", ...
                "Failed to generate DVB-S2 waveform with gen_dvbs2 for seed %s: %s", ...
                mat2str(seed), fallbackME.message);
        end
    end
end

function [waveform, fs, meta] = fallback_dvbs2_like(seed, fs, samplesPerSymbol)
    rng(seed);
    numSymbols = 4096;
    bits = randi([0 1], 2*numSymbols, 1);
    symbols = (2*bits(1:2:end)-1) + 1j*(2*bits(2:2:end)-1);
    symbols = symbols(:) / sqrt(2);
    waveform = repelem(symbols, samplesPerSymbol);
    waveform = waveform / sqrt(mean(abs(waveform).^2) + eps);

    meta = struct;
    meta.technology = "DVB-S2";
    meta.standard = "DVB-S2";
    meta.seed = seed;
    meta.generator = "fallback_dvbs2_like_qpsk";
    meta.sample_rate = fs;
    meta.modulation = "QPSK";
    meta.num_symbols = numSymbols;
    meta.samples_per_symbol = samplesPerSymbol;
end
