function [waveform, fs, meta] = gen_bluetooth(seed)
%GEN_BLUETOOTH Generate a conservative randomized Bluetooth test waveform.
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

        if exist('bluetoothTestWaveformConfig', 'file') == 2 && ...
                exist('bluetoothTestWaveform', 'file') == 2
            cfg = bluetoothTestWaveformConfig;
            cfg.Mode = 'LE1M';
            cfg.PayloadLength = randi([64 255]);
            cfg.SamplesPerSymbol = 8;

            waveform = bluetoothTestWaveform(cfg);
            waveform = waveform(:);
            fs = 1e6 * cfg.SamplesPerSymbol;

            meta = struct;
            meta.technology = "Bluetooth";
            meta.standard = "Bluetooth LE";
            meta.seed = seed;
            meta.generator = "bluetoothTestWaveform";
            meta.sample_rate = fs;
            meta.mode = string(cfg.Mode);
            meta.payload_length = cfg.PayloadLength;
            meta.modulation_index = cfg.ModulationIndex;
            meta.samples_per_symbol = cfg.SamplesPerSymbol;
        else
            [waveform, fs, meta] = fallback_gfsk_like(seed, 8e6, 8);
        end
    catch ME
        try
            [waveform, fs, meta] = fallback_gfsk_like(seed, 8e6, 8);
            meta.fallback_reason = string(ME.message);
        catch fallbackME
            error("gen_bluetooth:GenerationFailed", ...
                "Failed to generate Bluetooth waveform with gen_bluetooth for seed %s: %s", ...
                mat2str(seed), fallbackME.message);
        end
    end
end

function [waveform, fs, meta] = fallback_gfsk_like(seed, fs, samplesPerSymbol)
    rng(seed);
    numBits = 2048;
    bits = randi([0 1], numBits, 1);
    symbols = 2*bits - 1;
    phase = cumsum(repelem(symbols, samplesPerSymbol)) * (pi/8);
    waveform = exp(1j * phase(:));

    meta = struct;
    meta.technology = "Bluetooth";
    meta.standard = "Bluetooth LE";
    meta.seed = seed;
    meta.generator = "fallback_gfsk_like";
    meta.sample_rate = fs;
    meta.modulation = "GFSK-like";
    meta.num_bits = numBits;
    meta.samples_per_symbol = samplesPerSymbol;
end
