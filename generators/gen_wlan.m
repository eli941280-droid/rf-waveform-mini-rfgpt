function [waveform, fs, meta] = gen_wlan(seed)
%GEN_WLAN Generate a conservative randomized WLAN VHT waveform.
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

        channelBandwidthChoices = ["CBW20", "CBW40"];
        channelBandwidth = channelBandwidthChoices(randi(numel(channelBandwidthChoices)));
        mcs = randi([0 5]);
        apepLength = randi([512 2048]);

        cfg = wlanVHTConfig;
        cfg.ChannelBandwidth = channelBandwidth;
        cfg.MCS = mcs;
        cfg.APEPLength = apepLength;
        cfg.NumTransmitAntennas = 1;
        cfg.NumSpaceTimeStreams = 1;

        psdu = randi([0 1], cfg.PSDULength * 8, 1);
        waveform = wlanWaveformGenerator(psdu, cfg);
        waveform = waveform(:);

        fs = wlanSampleRate(cfg);

        meta = struct;
        meta.technology = "WLAN";
        meta.standard = "VHT";
        meta.seed = seed;
        meta.generator = "wlanWaveformGenerator";
        meta.sample_rate = fs;
        meta.channel_bandwidth = string(cfg.ChannelBandwidth);
        meta.mcs = cfg.MCS;
        meta.psdu_length = cfg.PSDULength;
        meta.apep_length = cfg.APEPLength;
        meta.num_transmit_antennas = cfg.NumTransmitAntennas;
        meta.num_space_time_streams = cfg.NumSpaceTimeStreams;
    catch ME
        error("gen_wlan:GenerationFailed", ...
            "Failed to generate WLAN waveform for seed %s: %s", ...
            mat2str(seed), ME.message);
    end
end
