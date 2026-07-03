function [y, freqMeta] = apply_random_frequency_shaping(x, seed)
%APPLY_RANDOM_FREQUENCY_SHAPING Apply randomized FFT-domain band shaping.
%
% Inputs:
%   x    : complex IQ waveform
%   seed : random seed passed to rng(seed)
%
% Outputs:
%   y        : frequency-shaped waveform with the same shape as x
%   freqMeta : metadata describing the selected frequency mask

    rng(seed);

    originalSize = size(x);
    xCol = x(:);
    numSamples = numel(xCol);

    X = fftshift(fft(xCol));
    normalizedFreq = ((0:numSamples-1).' - floor(numSamples / 2)) / numSamples;

    r = rand;
    if r < 0.20
        freqMode = "fullband";
    elseif r < 0.40
        freqMode = "narrowband";
    elseif r < 0.60
        freqMode = "low_shifted";
    elseif r < 0.80
        freqMode = "high_shifted";
    else
        freqMode = "two_subbands";
    end

    switch freqMode
        case "fullband"
            center = -0.04 + 0.08 * rand;
            width = 0.68 + 0.22 * rand;
            bandRanges = clipped_band(center, width);

        case "narrowband"
            center = -0.18 + 0.36 * rand;
            width = 0.14 + 0.22 * rand;
            bandRanges = clipped_band(center, width);

        case "low_shifted"
            center = -0.34 + 0.16 * rand;
            width = 0.18 + 0.20 * rand;
            bandRanges = clipped_band(center, width);

        case "high_shifted"
            center = 0.18 + 0.16 * rand;
            width = 0.18 + 0.20 * rand;
            bandRanges = clipped_band(center, width);

        case "two_subbands"
            lowBandStart = -0.49 + 0.14 * rand;
            lowBandEnd = lowBandStart + 0.08 + 0.16 * rand;
            highBandStart = 0.08 + 0.18 * rand;
            highBandEnd = highBandStart + 0.08 + 0.16 * rand;
            bandRanges = [
                max(-0.5, lowBandStart), min(-0.02, lowBandEnd)
                max(0.02, highBandStart), min(0.5, highBandEnd)
            ];
    end

    mask = zeros(numSamples, 1);
    for k = 1:size(bandRanges, 1)
        lowEdge = min(bandRanges(k, :));
        highEdge = max(bandRanges(k, :));
        bandRanges(k, :) = [lowEdge, highEdge];
        mask(normalizedFreq >= lowEdge & normalizedFreq <= highEdge) = 1;
    end

    % Simplified hard frequency shaping; this can later be replaced by a
    % smooth mask to reduce spectral edge artifacts.
    yCol = ifft(ifftshift(X .* mask));
    yCol = yCol / sqrt(mean(abs(yCol).^2) + eps);
    y = reshape(yCol, originalSize);

    freqMeta = struct;
    freqMeta.freq_mode = freqMode;
    freqMeta.active_band_fraction = mean(mask > 0);
    freqMeta.band_ranges = bandRanges;
end

function bandRange = clipped_band(center, width)
    halfWidth = width / 2;
    bandRange = [max(-0.5, center - halfWidth), min(0.5, center + halfWidth)];
end
