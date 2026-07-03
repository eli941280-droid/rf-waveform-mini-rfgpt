function [waveform, fs, meta] = gen_5g_nr(seed)
%GEN_5G_NR Generate a conservative randomized 5G NR waveform.
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

        if rand < 0.5
            cfg = nrDLCarrierConfig;
            linkDirection = "DL";
        else
            cfg = nrULCarrierConfig;
            linkDirection = "UL";
        end

        numSubframesMeta = "unknown";
        ncellidMeta = "unknown";
        channelBandwidthMeta = "unknown";
        subcarrierSpacingMeta = "unknown";

        if isprop(cfg, 'NumSubframes')
            cfg.NumSubframes = 10;
            numSubframesMeta = 10;
        end

        if isprop(cfg, 'FrequencyRange')
            cfg.FrequencyRange = "FR1";
        end

        if isprop(cfg, 'NCellID')
            cfg.NCellID = randi([0 1007]);
            ncellidMeta = cfg.NCellID;
        end

        if isprop(cfg, 'ChannelBandwidth')
            channelBandwidthChoices = [10 20];
            cfg.ChannelBandwidth = channelBandwidthChoices(randi(numel(channelBandwidthChoices)));
            channelBandwidthMeta = cfg.ChannelBandwidth;
        end

        if isprop(cfg, 'SubcarrierSpacing')
            subcarrierSpacingChoices = [15 30 60];
            cfg.SubcarrierSpacing = subcarrierSpacingChoices(randi(numel(subcarrierSpacingChoices)));
            subcarrierSpacingMeta = cfg.SubcarrierSpacing;
        end

        [waveform, info] = nrWaveformGenerator(cfg);
        waveform = waveform(:);

        fs = [];
        if isfield(info, 'SampleRate')
            fs = info.SampleRate;
        elseif isfield(info, 'ResourceGrids') && isstruct(info.ResourceGrids) && ...
                ~isempty(info.ResourceGrids) && ...
                isfield(info.ResourceGrids, 'Info')
            resourceGridInfo = info.ResourceGrids(1).Info;
            if isstruct(resourceGridInfo) && isfield(resourceGridInfo, 'SampleRate')
                fs = resourceGridInfo.SampleRate;
            end
        end

        if isempty(fs)
            error("gen_5g_nr:MissingSampleRate", ...
                "Cannot find SampleRate from nrWaveformGenerator info.");
        end

        meta = struct;
        meta.technology = "5G NR";
        meta.link_direction = linkDirection;
        meta.seed = seed;
        meta.generator = "nrWaveformGenerator";
        meta.sample_rate = fs;
        meta.num_subframes = numSubframesMeta;
        meta.ncellid = ncellidMeta;
        meta.channel_bandwidth = channelBandwidthMeta;
        meta.subcarrier_spacing = subcarrierSpacingMeta;
    catch ME
        error("gen_5g_nr:GenerationFailed", ...
            "Failed to generate 5G NR waveform for seed %s: %s", ...
            mat2str(seed), ME.message);
    end
end
