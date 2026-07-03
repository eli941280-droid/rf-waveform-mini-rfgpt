function [y, gatingMeta, freqMeta, profileMeta] = apply_technology_visual_profile(x, fs, technology, seed)
%APPLY_TECHNOLOGY_VISUAL_PROFILE Apply technology-aware lightweight shaping.
%
% Inputs:
%   x          : complex IQ waveform
%   fs         : sample rate in Hz
%   technology : technology label, such as "5G NR" or "WLAN"
%   seed       : random seed passed to rng(seed)
%
% Outputs:
%   y           : shaped waveform with the same shape as x
%   gatingMeta  : time occupancy metadata
%   freqMeta    : frequency occupancy metadata
%   profileMeta : technology visual profile metadata

    rng(seed);

    originalSize = size(x);
    xCol = x(:);
    numSamples = numel(xCol);
    tech = lower(strtrim(char(technology)));

    profileMeta = struct;
    profileMeta.technology_profile = string(technology);
    profileMeta.fs = fs;
    profileMeta.seed = seed;
    profileMeta.profile_generator = "apply_technology_visual_profile";

    switch tech
        case {'5g nr', 'nr', '5g'}
            profileMeta.profile_name = "wide_ofdm_grid_like";
            timeMode = weighted_choice(["full", "double_burst"], [0.85 0.15]);
            freqMode = "nr_wide_center";

        case {'lte'}
            profileMeta.profile_name = "lte_medium_wide_ofdm_like";
            timeMode = weighted_choice(["full", "single_burst"], [0.80 0.20]);
            freqMode = "lte_medium_center";

        case {'umts', 'wcdma'}
            profileMeta.profile_name = "continuous_wcdma_like";
            timeMode = "full";
            freqMode = "umts_mid_center";

        case {'wlan', 'wifi'}
            profileMeta.profile_name = "packet_burst_wlan_like";
            timeMode = weighted_choice(["wlan_double_burst", "wlan_periodic_burst"], [0.35 0.65]);
            freqMode = weighted_choice(["wlan_wide_low", "wlan_wide_high"], [0.50 0.50]);

        case {'dvb-s2', 'dvbs2', 'dvb'}
            profileMeta.profile_name = "continuous_single_carrier_like";
            timeMode = "full";
            freqMode = "dvbs2_narrow_center";

        case {'bluetooth', 'bt'}
            profileMeta.profile_name = "narrowband_burst_hopping_like";
            timeMode = weighted_choice(["bt_periodic_hops", "bt_dense_hops"], [0.70 0.30]);
            freqMode = "bluetooth_hopping";

        otherwise
            profileMeta.profile_name = "generic";
            timeMode = weighted_choice(["full", "single_burst", "double_burst", "periodic_burst"], [0.25 0.35 0.25 0.15]);
            freqMode = weighted_choice(["wide_center", "medium_center", "narrow_center"], [0.35 0.40 0.25]);
    end

    if ismember(string(timeMode), ["bt_periodic_hops", "bt_dense_hops"])
        xCol = repeat_to_min_length(xCol, 65536);
        originalSize = size(xCol);
    end

    [xCol, gatingMeta] = apply_time_profile(xCol, timeMode, seed + 11);
    if strcmp(char(freqMode), 'bluetooth_hopping')
        [xCol, freqMeta] = apply_bluetooth_hopping_profile(xCol, gatingMeta.burst_ranges, seed + 22);
    else
        [xCol, freqMeta] = apply_frequency_profile(xCol, freqMode, seed + 22);
    end

    y = reshape(xCol, originalSize);
end

function choice = weighted_choice(options, weights)
    weights = weights(:) / sum(weights);
    edges = cumsum(weights);
    idx = find(rand <= edges, 1, 'first');
    choice = options(idx);
end

function [yCol, gatingMeta] = apply_time_profile(xCol, timeMode, seed)
    rng(seed);

    numSamples = numel(xCol);
    leakageGain = 0.01;
    gate = leakageGain * ones(numSamples, 1);
    timeRollSamples = randi([0, max(numSamples - 1, 0)]);
    xCol = circshift(xCol(:), timeRollSamples);

    burstRanges = zeros(0, 2);

    switch char(timeMode)
        case 'full'
            burstRanges = [1 numSamples];

        case 'single_burst'
            burstLen = random_fraction_length(numSamples, 0.30, 0.85);
            startIdx = randi([1, numSamples - burstLen + 1]);
            burstRanges = [startIdx, startIdx + burstLen - 1];

        case 'wlan_single_burst'
            burstLen = random_fraction_length(numSamples, 0.12, 0.38);
            startIdx = randi([1, numSamples - burstLen + 1]);
            burstRanges = [startIdx, startIdx + burstLen - 1];

        case 'bt_single_hop'
            burstLen = random_fraction_length(numSamples, 0.04, 0.12);
            startIdx = randi([1, numSamples - burstLen + 1]);
            burstRanges = [startIdx, startIdx + burstLen - 1];

        case 'double_burst'
            burstLen1 = random_fraction_length(numSamples, 0.16, 0.34);
            burstLen2 = random_fraction_length(numSamples, 0.16, 0.34);
            midIdx = floor(numSamples / 2);
            start1Max = max(1, midIdx - burstLen1 + 1);
            startIdx1 = randi([1, start1Max]);
            start2Min = min(numSamples, midIdx + 1);
            start2Max = max(start2Min, numSamples - burstLen2 + 1);
            startIdx2 = randi([start2Min, start2Max]);
            burstRanges = [
                startIdx1, startIdx1 + burstLen1 - 1
                startIdx2, startIdx2 + burstLen2 - 1
            ];

        case 'wlan_double_burst'
            burstLen1 = random_fraction_length(numSamples, 0.05, 0.13);
            burstLen2 = random_fraction_length(numSamples, 0.05, 0.13);
            midIdx = floor(numSamples / 2);
            start1Max = max(1, midIdx - burstLen1 + 1);
            startIdx1 = randi([1, start1Max]);
            start2Min = min(numSamples, midIdx + 1);
            start2Max = max(start2Min, numSamples - burstLen2 + 1);
            startIdx2 = randi([start2Min, start2Max]);
            burstRanges = [
                startIdx1, startIdx1 + burstLen1 - 1
                startIdx2, startIdx2 + burstLen2 - 1
            ];

        case 'bt_double_hops'
            burstLen1 = random_fraction_length(numSamples, 0.025, 0.075);
            burstLen2 = random_fraction_length(numSamples, 0.025, 0.075);
            midIdx = floor(numSamples / 2);
            start1Max = max(1, midIdx - burstLen1 + 1);
            startIdx1 = randi([1, start1Max]);
            start2Min = min(numSamples, midIdx + 1);
            start2Max = max(start2Min, numSamples - burstLen2 + 1);
            startIdx2 = randi([start2Min, start2Max]);
            burstRanges = [
                startIdx1, startIdx1 + burstLen1 - 1
                startIdx2, startIdx2 + burstLen2 - 1
            ];

        case 'periodic_burst'
            numBursts = randi([4 12]);
            periodLen = max(1, floor(numSamples / numBursts));
            jitterMax = max(0, floor(0.20 * periodLen));
            burstRanges = zeros(numBursts, 2);
            for k = 1:numBursts
                burstLen = random_fraction_length(numSamples, 0.025, 0.075);
                nominalStart = (k - 1) * periodLen + 1;
                if jitterMax > 0
                    startIdx = nominalStart + randi([-jitterMax jitterMax]);
                else
                    startIdx = nominalStart;
                end
                startIdx = min(max(1, startIdx), numSamples - burstLen + 1);
                burstRanges(k, :) = [startIdx, startIdx + burstLen - 1];
            end

        case 'wlan_periodic_burst'
            numBursts = randi([4 7]);
            periodLen = max(1, floor(numSamples / numBursts));
            jitterMax = max(0, floor(0.25 * periodLen));
            burstRanges = zeros(numBursts, 2);
            for k = 1:numBursts
                burstLen = random_fraction_length(numSamples, 0.018, 0.045);
                nominalStart = (k - 1) * periodLen + 1;
                if jitterMax > 0
                    startIdx = nominalStart + randi([-jitterMax jitterMax]);
                else
                    startIdx = nominalStart;
                end
                startIdx = min(max(1, startIdx), numSamples - burstLen + 1);
                burstRanges(k, :) = [startIdx, startIdx + burstLen - 1];
            end

        case 'bt_periodic_hops'
            numBursts = randi([18 32]);
            periodLen = max(1, floor(numSamples / numBursts));
            jitterMax = max(0, floor(0.20 * periodLen));
            burstRanges = zeros(numBursts, 2);
            for k = 1:numBursts
                burstLen = random_fraction_length(numSamples, 0.006, 0.016);
                nominalStart = (k - 1) * periodLen + 1;
                if jitterMax > 0
                    startIdx = nominalStart + randi([-jitterMax jitterMax]);
                else
                    startIdx = nominalStart;
                end
                startIdx = min(max(1, startIdx), numSamples - burstLen + 1);
                burstRanges(k, :) = [startIdx, startIdx + burstLen - 1];
            end

        case 'bt_dense_hops'
            numBursts = randi([34 52]);
            periodLen = max(1, floor(numSamples / numBursts));
            jitterMax = max(0, floor(0.15 * periodLen));
            burstRanges = zeros(numBursts, 2);
            for k = 1:numBursts
                burstLen = random_fraction_length(numSamples, 0.0035, 0.010);
                nominalStart = (k - 1) * periodLen + 1;
                if jitterMax > 0
                    startIdx = nominalStart + randi([-jitterMax jitterMax]);
                else
                    startIdx = nominalStart;
                end
                startIdx = min(max(1, startIdx), numSamples - burstLen + 1);
                burstRanges(k, :) = [startIdx, startIdx + burstLen - 1];
            end
    end

    for k = 1:size(burstRanges, 1)
        startIdx = max(1, burstRanges(k, 1));
        endIdx = min(numSamples, burstRanges(k, 2));
        gate(startIdx:endIdx) = 1;
        burstRanges(k, :) = [startIdx, endIdx];
    end

    yCol = xCol .* gate;

    gatingMeta = struct;
    gatingMeta.gating_mode = canonical_gating_mode(timeMode);
    gatingMeta.time_profile_mode = string(timeMode);
    gatingMeta.active_fraction = mean(gate > leakageGain);
    gatingMeta.burst_ranges = burstRanges;
    gatingMeta.leakage_gain = leakageGain;
    gatingMeta.time_roll_samples = timeRollSamples;
end

function [yCol, freqMeta] = apply_frequency_profile(xCol, freqMode, seed)
    rng(seed);

    numSamples = numel(xCol);
    X = fftshift(fft(xCol(:)));
    normalizedFreq = ((0:numSamples-1).' - floor(numSamples / 2)) / numSamples;

    switch char(freqMode)
        case 'nr_wide_center'
            bandRanges = random_band(0, 0.86, 0.98, 0.015);

        case 'lte_medium_center'
            bandRanges = random_band(0, 0.48, 0.62, 0.025);
        case 'lte_medium_low'
            bandRanges = random_band(-0.14, 0.42, 0.56, 0.025);
        case 'lte_medium_high'
            bandRanges = random_band(0.14, 0.42, 0.56, 0.025);

        case 'umts_mid_center'
            bandRanges = random_band(0, 0.30, 0.42, 0.020);
        case 'umts_mid_low'
            bandRanges = random_band(-0.12, 0.28, 0.38, 0.020);
        case 'umts_mid_high'
            bandRanges = random_band(0.12, 0.28, 0.38, 0.020);

        case 'wlan_wide_center'
            bandRanges = random_band(0, 0.58, 0.74, 0.040);
        case 'wlan_wide_low'
            bandRanges = random_band(-0.25, 0.34, 0.46, 0.020);
        case 'wlan_wide_high'
            bandRanges = random_band(0.25, 0.34, 0.46, 0.020);

        case 'dvbs2_narrow_center'
            bandRanges = random_band(0, 0.10, 0.18, 0.025);
        case 'dvbs2_narrow_low'
            bandRanges = random_band(-0.22, 0.09, 0.16, 0.025);
        case 'dvbs2_narrow_high'
            bandRanges = random_band(0.22, 0.09, 0.16, 0.025);

        case 'bt_base_narrow'
            bandRanges = random_band(0, 0.035, 0.070, 0.010);

        case 'wide_center'
            bandRanges = random_band(0, 0.72, 0.90, 0.04);
        case 'wide_low'
            bandRanges = random_band(-0.10, 0.62, 0.82, 0.04);
        case 'wide_high'
            bandRanges = random_band(0.10, 0.62, 0.82, 0.04);

        case 'medium_center'
            bandRanges = random_band(0, 0.38, 0.58, 0.08);
        case 'medium_low'
            bandRanges = random_band(-0.16, 0.32, 0.52, 0.05);
        case 'medium_high'
            bandRanges = random_band(0.16, 0.32, 0.52, 0.05);

        case 'umts_center'
            bandRanges = random_band(0, 0.48, 0.68, 0.04);
        case 'umts_low'
            bandRanges = random_band(-0.08, 0.44, 0.62, 0.04);
        case 'umts_high'
            bandRanges = random_band(0.08, 0.44, 0.62, 0.04);

        case 'narrow_center'
            bandRanges = random_band(0, 0.18, 0.30, 0.04);
        case 'narrow_low'
            bandRanges = random_band(-0.20, 0.14, 0.26, 0.04);
        case 'narrow_high'
            bandRanges = random_band(0.20, 0.14, 0.26, 0.04);

        case 'very_narrow_low'
            bandRanges = random_band(-0.24, 0.06, 0.14, 0.08);
        case 'very_narrow_high'
            bandRanges = random_band(0.24, 0.06, 0.14, 0.08);
        case 'two_tiny_subbands'
            bandRanges = [
                random_band(-0.26, 0.05, 0.11, 0.05)
                random_band(0.22, 0.05, 0.11, 0.05)
            ];
        otherwise
            bandRanges = random_band(0, 0.35, 0.65, 0.08);
    end

    mask = zeros(numSamples, 1);
    for k = 1:size(bandRanges, 1)
        lowEdge = min(bandRanges(k, :));
        highEdge = max(bandRanges(k, :));
        bandRanges(k, :) = [lowEdge, highEdge];
        mask(normalizedFreq >= lowEdge & normalizedFreq <= highEdge) = 1;
    end

    yCol = ifft(ifftshift(X .* mask));
    yCol = yCol / sqrt(mean(abs(yCol).^2) + eps);

    freqMeta = struct;
    freqMeta.freq_mode = string(freqMode);
    freqMeta.active_band_fraction = mean(mask > 0);
    freqMeta.band_ranges = bandRanges;
end

function [yCol, freqMeta] = apply_bluetooth_hopping_profile(xCol, burstRanges, seed)
    rng(seed);

    numSamples = numel(xCol);
    [baseCol, baseMeta] = apply_frequency_profile(xCol, "bt_base_narrow", seed + 1);
    yCol = 0.01 * baseCol;
    n = (0:numSamples-1).';

    hopCenters = [-0.44 -0.31 -0.18 0.18 0.31 0.44];
    hopWidth = 0.035;
    bandRanges = zeros(size(burstRanges, 1), 2);
    hopOffset = randi(numel(hopCenters));

    for k = 1:size(burstRanges, 1)
        startIdx = max(1, burstRanges(k, 1));
        endIdx = min(numSamples, burstRanges(k, 2));
        centerIdx = mod(k + hopOffset - 2, numel(hopCenters)) + 1;
        center = hopCenters(centerIdx) + 0.006 * (2*rand - 1);
        phase = exp(1j * 2*pi * center * n(startIdx:endIdx));
        yCol(startIdx:endIdx) = baseCol(startIdx:endIdx) .* phase;
        bandRanges(k, :) = [max(-0.5, center - hopWidth/2), min(0.5, center + hopWidth/2)];
    end

    yCol = yCol / sqrt(mean(abs(yCol).^2) + eps);

    freqMeta = struct;
    freqMeta.freq_mode = "bluetooth_hopping";
    freqMeta.active_band_fraction = min(1, size(bandRanges, 1) * hopWidth);
    freqMeta.band_ranges = bandRanges;
    freqMeta.base_band_ranges = baseMeta.band_ranges;
end

function gatingMode = canonical_gating_mode(timeMode)
    switch char(timeMode)
        case {'wlan_single_burst', 'bt_single_hop'}
            gatingMode = "single_burst";
        case {'wlan_double_burst', 'bt_double_hops'}
            gatingMode = "double_burst";
        case {'wlan_periodic_burst', 'bt_periodic_hops', 'bt_dense_hops'}
            gatingMode = "periodic_burst";
        otherwise
            gatingMode = string(timeMode);
    end
end

function yCol = repeat_to_min_length(xCol, minLength)
    xCol = xCol(:);
    if numel(xCol) >= minLength
        yCol = xCol;
        return;
    end
    repeatCount = ceil(minLength / numel(xCol));
    yCol = repmat(xCol, repeatCount, 1);
    yCol = yCol(1:minLength);
end

function bandRange = random_band(centerNominal, minWidth, maxWidth, centerJitter)
    center = centerNominal + centerJitter * (2*rand - 1);
    width = minWidth + (maxWidth - minWidth) * rand;
    halfWidth = width / 2;
    bandRange = [max(-0.5, center - halfWidth), min(0.5, center + halfWidth)];
end

function burstLen = random_fraction_length(numSamples, minFraction, maxFraction)
    burstLen = round(numSamples * (minFraction + (maxFraction - minFraction) * rand));
    burstLen = min(max(1, burstLen), numSamples);
end
