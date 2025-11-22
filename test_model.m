%% ==============================================
%  TESTING TRAINED SEISMIC MODEL ON NEW SIGNALS
%  ==============================================
clear; clc; close all;
fprintf('\n🌋 Testing Trained Seismic Model...\n');
% --- Load Trained Model ---
load('seismic_model.mat', 'SVM', 'cfg');
fprintf('Loaded trained model: %s\n', cfg.model_file);
% --- Generate new synthetic test data ---
testData = generate_synthetic(cfg);
d = testData{1};  % pick first test file
fprintf('Testing on: %s\n', d.filename);
% --- Preprocess and Bandpass Filter ---
[t, x] = preprocess(d.time, d.acc, cfg.fs);
x_bp = bandpass_filter(x, cfg.bp_low, cfg.bp_high, cfg.fs);
% --- Detect Events (STA/LTA) ---
ev = detect_events_sta_lta(x_bp, cfg, cfg.fs);
fprintf('Detected %d candidate events.\n', size(ev,1));

% --- Extract Features for Each Event ---
[F, ~] = extract_features(x_bp, ev, cfg);

% --- FIX: Check if features were extracted ---
if isempty(F)
    fprintf('\nNo features extracted for prediction.\n');
    fprintf('Prediction skipped. Consider adjusting STA/LTA thresholds (cfg.sta_on, cfg.sta_off) or checking data.\n');
    return;
end

% --- Predict using the trained SVM model ---
[labels, scores] = predict(SVM, F);

% --- Display results in a table ---
fprintf('\nPrediction Results:\n');
T = table((1:length(labels))', labels, scores(:,2), ...
    'VariableNames', {'Event_ID', 'PredictedLabel', 'Probability'});
disp(T);
% --- Visualize waveform and detected events ---
figure;
plot(t, x_bp, 'b'); hold on;
for i = 1:size(ev,1)
    xline(ev(i,1), 'r--', 'LineWidth', 1);
    xline(ev(i,2), 'g--', 'LineWidth', 1);
end
xlabel('Time (s)'); ylabel('Amplitude');
title(['Detected Events in ', d.filename]);
legend('Signal','Event Start','Event End');
grid on;