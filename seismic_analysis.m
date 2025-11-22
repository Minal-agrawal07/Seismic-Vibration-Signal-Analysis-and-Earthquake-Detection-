
clear; close all; clc;
fprintf('\n🌋 Seismic Vibration Signal Analysis Started...\n');

%% ---------------- Configuration ----------------
cfg.mode = 'demo';             % 'demo' or 'real'
cfg.fs = 100;                  % Sampling frequency (Hz)
cfg.bp_low = 0.5;              % Bandpass filter low cutoff (Hz)
cfg.bp_high = 20;              % Bandpass filter high cutoff (Hz)
cfg.sta_win = 1.0;             % STA window (s)
cfg.lta_win = 10.0;            % LTA window (s)
cfg.sta_on = 3.0;              % STA/LTA trigger ratio
cfg.sta_off = 1.5;             % STA/LTA de-trigger ratio
cfg.feature_win = 5.0;         % Feature window (s)
cfg.min_event_len = 1.0;       % Minimum event length (s)
cfg.model_file = 'seismic_model.mat';

if strcmp(cfg.mode, 'demo')
    % Generate synthetic dataset
    data = generate_synthetic(cfg);
else
    % Load real dataset from "data" folder
    data = load_real_data('data', cfg.fs);
end


%% ---------------- Pipeline ----------------
allF = [];
allY = [];

for f = 1:numel(data)
    fprintf('\nProcessing %s ...\n', data{f}.filename);

    % Preprocessing
    [t, x] = preprocess(data{f}.time, data{f}.acc, cfg.fs);
    x_bp = bandpass_filter(x, cfg.bp_low, cfg.bp_high, cfg.fs);

  

    % Event detection using STA/LTA
    ev = detect_events_sta_lta(x_bp, cfg, cfg.fs);
    fprintf('  → %d candidate events detected\n', size(ev, 1));

    % Feature extraction
    [F, Y] = extract_features(x_bp, ev, cfg);
    allF = [allF; F];
    allY = [allY; Y];
end

if isempty(allF)
    error('No features extracted. Increase sensitivity or check data.');
end

% Model Training 
fprintf('\nTraining model on %d samples...\n', size(allF, 1));
SVM = fitcsvm(allF, allY, 'KernelFunction', 'rbf', 'Standardize', true);
SVM = fitPosterior(SVM);
save(cfg.model_file, 'SVM', 'cfg');
fprintf('Model saved: %s\n', cfg.model_file);

%Evaluation 
cv = cvpartition(allY, 'KFold', 5);
Ypred = zeros(size(allY));
scores = zeros(size(allY));

for i = 1:cv.NumTestSets
    tr = cv.training(i);
    te = cv.test(i);
    M = fitPosterior(fitcsvm(allF(tr, :), allY(tr), ...
        'KernelFunction', 'rbf', 'Standardize', true));
    [lbl, sc] = predict(M, allF(te, :));
    Ypred(te) = lbl;
    scores(te) = sc(:, 2);
end

acc = mean(Ypred == allY);
cm = confusionmat(allY, Ypred);
fprintf('\nAccuracy: %.2f%%\n', acc * 100);
disp(array2table(cm, 'VariableNames', {'Pred0', 'Pred1'}, ...
    'RowNames', {'True0', 'True1'}));

[Xroc, Yroc, ~, AUC] = perfcurve(allY, scores, 1);
fprintf('AUC = %.3f\n', AUC);

figure;
plot(Xroc, Yroc);
xlabel('FPR');
ylabel('TPR');
title(sprintf('ROC (AUC=%.3f)', AUC));
grid on;

fprintf('\n✅ Analysis Complete.\n');

%helper funcn

function data = generate_synthetic(cfg)
    fs = cfg.fs;
    dur = 180;
    n = 8;        % number of files
    data = {};

    for i = 1:n
        t = (0:1/fs:dur-1/fs)';
        noise = 0.02 * randn(size(t));
        acc = noise;

        nq = randi([2, 4]);  % number of events per file
        for j = 1:nq
            len = 2 + 4 * rand();
            st = 5 + (dur - 10 - len) * rand();
            i1 = round(st * fs) + 1;
            i2 = min(numel(t), i1 + round(len * fs));

            tt = (0:(i2 - i1))' / fs;
            f0 = 3 + 6 * rand();
            wave = (1 - 2 * (pi * f0 * tt).^2) .* exp(-(pi * f0 * tt).^2);

            acc(i1:i2) = acc(i1:i2) + 0.5 * (wave + 0.3 * randn(size(wave)));
        end

        data{end + 1} = struct('filename', sprintf('synthetic_%d', i), ...
            'time', t, 'acc', acc, 'fs', fs);
    end
end

function [t, x] = preprocess(ti, xi, fs)
    xi = xi(:) - mean(xi);
    t = (0:numel(xi)-1)' / fs;
    [b, a] = butter(2, 0.1 / (fs / 2), 'high');
    x = filtfilt(b, a, xi);
end

function xbp = bandpass_filter(x, fl, fh, fs)
    [b, a] = butter(4, [fl fh] / (fs / 2), 'bandpass');
    xbp = filtfilt(b, a, x);
end



function ev = detect_events_sta_lta(x, cfg, fs)
    staN = max(1, round(cfg.sta_win * fs));
    ltaN = max(1, round(cfg.lta_win * fs));
    sta = movmean(x.^2, staN);
    lta = movmean(x.^2, ltaN) + eps;
    r = sta ./ lta;

    thr_on = cfg.sta_on;
    thr_off = cfg.sta_off;

    idx = find(r > thr_on);
    if isempty(idx)
        ev = zeros(0, 2);
        return;
    end

    gaps = find(diff(idx) > 1);
    starts = [idx(1); idx(gaps + 1)];
    ends = [idx(gaps); idx(end)];
    res = [];

    for i = 1:numel(starts)
        s = starts(i);
        e = ends(i);

        while s > 1 && r(s) > thr_off
            s = s - 1;
        end
        while e < numel(r) && r(e) > thr_off
            e = e + 1;
        end

        if (e - s) / fs >= cfg.min_event_len
            res = [res; s, e];
        end
    end

    ev = res / fs;
end

function [F, Y] = extract_features(x, ev, cfg)
    fs = cfg.fs;
    win = round(cfg.feature_win * fs);
    N = numel(x);
    F = [];
    Y = [];

    % Positive samples (detected events)
    for i = 1:size(ev, 1)
        c = round(mean(ev(i, :)) * fs);
        s = max(1, c - floor(win / 2));
        e = min(N, s + win - 1);
        seg = x(s:e);
        F = [F; features(seg, fs)];
        Y = [Y; 1];
    end

    % Negative samples (noise segments)
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

    % Time-domain features
    rmsv = sqrt(mean(seg.^2));
    peak = max(abs(seg));
    skew = skewness(seg);
    kurt = kurtosis(seg);

    % Frequency-domain features
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

function data = load_real_data(folder, fs)
    files = dir(fullfile(folder, '*.csv'));
    data = {};

    for i = 1:numel(files)
        T = readtable(fullfile(folder, files(i).name));
        t = T{:, 1};
        x = T{:, 2};
        data{end + 1} = struct('filename', files(i).name, ...
            'time', t, 'acc', x, 'fs', fs);
    end
end
