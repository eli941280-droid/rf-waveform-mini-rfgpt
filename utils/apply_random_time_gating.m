function [y, gatingMeta] = apply_random_time_gating(x, seed)
%APPLY_RANDOM_TIME_GATING Apply randomized time-domain burst gating.
%
% Inputs:
%   x    : complex IQ waveform
%   seed : random seed passed to rng(seed)
%
% Outputs:
%   y          : gated waveform with the same shape as x
%   gatingMeta : metadata describing the selected gating pattern

    rng(seed);

    originalSize = size(x);
    xCol = x(:);
    numSamples = numel(xCol);
    leakageGain = 0.02;
    gate = leakageGain * ones(numSamples, 1);
    timeRollSamples = randi([0, max(numSamples - 1, 0)]);
    xCol = circshift(xCol, timeRollSamples);

    r = rand;
    if r < 0.25
        gatingMode = "full";
    elseif r < 0.60
        gatingMode = "single_burst";
    elseif r < 0.85
        gatingMode = "double_burst";
    else
        gatingMode = "periodic_burst";
    end

    burstRanges = zeros(0, 2);

    switch gatingMode
        case "full"
            burstRanges = [1 numSamples];

        case "single_burst"
            burstLen = random_fraction_length(numSamples, 0.30, 0.80);
            startIdx = randi([1, numSamples - burstLen + 1]);
            burstRanges = [startIdx, startIdx + burstLen - 1];

        case "double_burst"
            midIdx = floor(numSamples / 2);
            burstLen1 = random_fraction_length(numSamples, 0.15, 0.35);
            burstLen2 = random_fraction_length(numSamples, 0.15, 0.35);

            start1Max = max(1, midIdx - burstLen1 + 1);
            startIdx1 = randi([1, start1Max]);

            start2Min = min(numSamples, midIdx + 1);
            start2Max = max(start2Min, numSamples - burstLen2 + 1);
            startIdx2 = randi([start2Min, start2Max]);

            burstRanges = [
                startIdx1, startIdx1 + burstLen1 - 1
                startIdx2, startIdx2 + burstLen2 - 1
            ];

        case "periodic_burst"
            numBursts = randi([4 10]);
            periodLen = max(1, floor(numSamples / numBursts));
            jitterMax = max(0, floor(0.25 * periodLen));
            burstRanges = zeros(numBursts, 2);

            for k = 1:numBursts
                burstLen = random_fraction_length(numSamples, 0.03, 0.08);
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
    y = reshape(yCol, originalSize);

    gatingMeta = struct;
    gatingMeta.gating_mode = gatingMode;
    gatingMeta.active_fraction = mean(gate > leakageGain);
    gatingMeta.burst_ranges = burstRanges;
    gatingMeta.time_roll_samples = timeRollSamples;
end

function burstLen = random_fraction_length(numSamples, minFraction, maxFraction)
    burstLen = round(numSamples * (minFraction + (maxFraction - minFraction) * rand));
    burstLen = min(max(1, burstLen), numSamples);
end
