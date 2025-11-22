# Seismic-Vibration-Signal-Analysis-and-Earthquake-Detection-

Overview

This project performs synthetic seismic vibration signal analysis to detect earthquake-like events using signal processing techniques. It simulates seismic signals, adds controlled noise, inserts events, applies filtering, computes signal energy, and automatically detects event boundaries.
The workflow produces visual results including identified event start and end points, as shown in the final plotted graph.

Features

Synthetic seismic signal generation
Noise injection to replicate real-world conditions
Event embedding with adjustable intensity and duration
Signal smoothing and band-pass filtering
Short-Time Energy based event detection
Automatic marking of event start and end times
Visualization of detected activity on the waveform


Methodology

1. Signal Generation
A synthetic baseline signal is created with controlled frequency and amplitude ranges. Random noise is added to simulate natural vibrations.
2. Event Simulation
High‑amplitude pulses are injected at known timestamps to act as earthquake events.
3. Pre‑processing
Noise reduction
Smoothing
Filtering for relevant frequency components
4. Event Detection
A short‑time energy thresholding method is used:
Compute sliding‑window energy
Compare with a dynamic threshold
Mark rising edge as event start
Mark falling edge as event end
5. Visualization
The final result plots:
Blue: processed seismic signal
Red dashed lines: detected event start times
Green dashed lines: detected event end times


Requirements:
MATLAB


Final Results:







<img width="657" height="552" alt="image" src="https://github.com/user-attachments/assets/37dc5b9e-6f51-4168-815f-0caf94be3dc6" />


License
This project is open for academic and learning purposes. Modify and extend as needed.
