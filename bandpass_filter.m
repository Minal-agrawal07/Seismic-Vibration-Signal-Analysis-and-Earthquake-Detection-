function xbp = bandpass_filter(x, fl, fh, fs)
    [b, a] = butter(4, [fl fh] / (fs / 2), 'bandpass');
    xbp = filtfilt(b, a, x);
end
