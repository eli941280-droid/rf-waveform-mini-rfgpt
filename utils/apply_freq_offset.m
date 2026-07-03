function y = apply_freq_offset(x, fs, freq_offset_hz)
%APPLY_FREQ_OFFSET Apply a carrier frequency offset to complex IQ samples.
%
% Inputs:
%   x              : complex IQ signal as a row or column vector
%   fs             : sample rate in Hz, must be non-empty and positive
%   freq_offset_hz : frequency offset in Hz
%
% Outputs:
%   y : frequency-shifted IQ signal as a column vector

    if isempty(fs) || ~isscalar(fs) || fs <= 0
        error("apply_freq_offset:InvalidSampleRate", ...
            "fs must be a non-empty positive scalar.");
    end

    x = x(:);
    n = (0:numel(x)-1).';

    y = x .* exp(1j * 2 * pi * freq_offset_hz .* n / fs);
end
