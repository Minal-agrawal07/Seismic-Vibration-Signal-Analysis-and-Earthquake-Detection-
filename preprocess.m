function [t, x] = preprocess(ti, xi, fs)
    xi = xi(:) - mean(xi);
    t = (0:numel(xi)-1)' / fs;
    [b, a] = butter(2, 0.1 / (fs / 2), 'high');
    x = filtfilt(b, a, xi);
end
