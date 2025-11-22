function [F, Y] = extract_features(x, ev, cfg)
    fs = cfg.fs;
    win = round(cfg.feature_win * fs);
    N = numel(x);
    F = [];
    Y = [];

    % Positive samples
    for i = 1:size(ev, 1)
        c = round(mean(ev(i, :)) * fs);
        s = max(1, c - floor(win / 2));
        e = min(N, s + win - 1);
        seg = x(s:e);
        F = [F; features(seg, fs)];
        Y = [Y; 1];
    end

    % Negative samples
    for i = 1:size(ev, 1)
        s = randi([1, N - win]);
        e = s + win - 1;
        seg = x(s:e);
        F = [F; features(seg, fs)];
        Y = [Y; 0];
    end
end

function f = features(seg, fs)
    seg = seg - mean(seg);
    N = numel(seg);

    rmsv = sqrt(mean(seg.^2));
    peak = max(abs(seg));
    skew = skewness(seg);
    kurt = kurtosis(seg);

    Y = fft(seg);
    P = abs(Y / N);
    P = P(1:N/2 + 1);
    P(2:end - 1) = 2 * P(2:end - 1);
    f_axis = (0:length(P) - 1)' * fs / N;

    [~, ix] = max(P);
    dom = f_axis(ix);
    cent = sum(f_axis .* P) / sum(P);
    bw = sqrt(sum(((f_axis - cent).^2) .* P) / sum(P));
    p = P / sum(P);
    ent = -sum(p .* log2(p + eps));

    f = [rmsv, peak, skew, kurt, dom, cent, bw, ent];
end
