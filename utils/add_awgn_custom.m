function y = add_awgn_custom(x, snr_db, seed)
%ADD_AWGN_CUSTOM Add complex Gaussian white noise to baseband IQ samples.
%
% Inputs:
%   x      : complex baseband IQ signal
%   snr_db : target signal-to-noise ratio in dB
%   seed   : optional random seed passed to rng(seed)
%
% Outputs:
%   y : noisy IQ signal with the same shape as x

    if nargin >= 3 && ~isempty(seed)
        rng(seed);
    end

    signalPower = mean(abs(x(:)).^2);
    noisePower = signalPower / (10^(snr_db / 10));

    noise = sqrt(noisePower / 2) .* ...
        (randn(size(x)) + 1j * randn(size(x)));

    y = x + noise;
end
