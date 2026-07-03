function [specDb, img] = make_spectrogram_image(x, fs, outPngPath, dynamicRangeDb)
%MAKE_SPECTROGRAM_IMAGE Convert complex IQ samples to dB spectrogram image.
%
% Inputs:
%   x              : complex baseband IQ waveform
%   fs             : sample rate in Hz
%   outPngPath     : output PNG path
%   dynamicRangeDb : optional dB range for clipping, default is 60
%                    Smaller values increase contrast but may lose weak signals.
%
% Outputs:
%   specDb : clipped dB spectrogram
%   img    : normalized 512x512 image in [0,1]

    if nargin < 4 || isempty(dynamicRangeDb)
        dynamicRangeDb = 60;
    end

    x = x(:);

    nfft = 512;
    winLen = 512;
    hop = 512;
    noverlap = 0;

    win = blackman(winLen, "periodic");

    [S, ~, ~] = spectrogram(x, win, noverlap, nfft, fs, "centered");

    powerSpec = abs(S).^2;
    rawDb = 10*log10(powerSpec + eps);

    maxDb = max(rawDb(:));
    minDb = maxDb - dynamicRangeDb;

    specDb = min(max(rawDb, minDb), maxDb);

    img = (specDb - minDb) / (maxDb - minDb + eps);
    img = imresize(img, [512 512]);

    [folder, ~, ~] = fileparts(outPngPath);
    if ~isempty(folder) && ~exist(folder, "dir")
        mkdir(folder);
    end

    imwrite(img, outPngPath);
end
