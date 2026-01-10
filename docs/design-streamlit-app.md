# Edge PDM Streamlit Dashboard — Design Spec

## Overview
A single-page interactive dashboard built with Streamlit that demonstrates the **entire Edge PDM pipeline** (signal → FFT → scaler → TFLite inference → anomaly detection) in real-time. Serves as a visual proof-of-concept for the ESP32 firmware, usable in interviews / CV demos.

## Stack
- **Framework:** Streamlit (pure Python)
- **Styling:** Custom CSS overrides (dark premium theme), full-width layout, no Streamlit default chrome
- **Charts:** Plotly (interactive, GPU-accelerated)
- **Font:** Geist (loaded via CDN)
- **Deploy:** Streamlit Cloud (free, permanent link)

## Layout (3-column bento grid)

```
┌─────────────────────────────────────────────────────┐
│  HEADER: "Edge PDM · Predictive Maintenance System" │
├────────────┬────────────────────────┬───────────────┤
│ CONTROLS   │  VISUALIZATION         │  STATUS       │
│ (20%)      │  (55%)                 │  (25%)        │
│            │                        │               │
│ Signal Type│  ┌──────────────────┐  │  ● LED Alert  │
│  ○ Normal  │  │  Time Domain    │  │  GREEN/RED    │
│  ○ Anomaly │  │  (waveform)     │  │               │
│            │  └──────────────────┘  │  Error: 0.023 │
│ Freq 1 ────│                        │  ━━●━━━━━━━   │
│ Freq 2 ────│  ┌──────────────────┐  │  Threshold    │
│ Freq 3 ────│  │  FFT Spectrum   │  │               │
│            │  │  (bar chart)     │  │  Anomalies    │
│ Noise ─────│  └──────────────────┘  │  2 / 50       │
│            │                        │               │
│            │                        │  Error History │
│            │                        │  ┌──────────┐ │
│            │                        │  │ mini-line│ │
│            │                        │  └──────────┘ │
└────────────┴────────────────────────┴───────────────┘
│  FOOTER: "Inference: 87ms · Model: 24.9KB · 95% Acc" │
└─────────────────────────────────────────────────────┘
```

## Components

### 1. Header
- Title + subtitle, glass-effect floating bar
- GitHub link / docs link in corner

### 2. Controls Panel
- **Signal type:** Segmented button (Normal / Anomaly) not a radio
- **Frequency sliders:** 3 sliders for harmonic amplitudes (10Hz, 25Hz, 50Hz for normal; extra 150Hz, 200Hz for anomaly)
- **Noise slider:** 0.0 – 0.5
- **Auto-animate toggle:** Continuous signal regeneration or step-by-step

### 3. Time Domain Plot (Plotly)
- Yellow/white line on dark background
- 128 samples at 1kHz
- Smooth line, no markers, subtle glow effect

### 4. FFT Spectrum (Plotly)
- Bar chart, 64 frequency bins
- Gradient bar color (low freq = cool blue, high freq = warm red)
- Dynamic Y-axis

### 5. Status Panel
- **LED indicator:** Circular div, CSS glow animation. Green = normal, Red (pulsing) = anomaly
- **Reconstruction error:** Large monospace number + horizontal progress bar filling toward threshold
- **Anomaly stats:** "X / Y detected" + percentage

### 6. Error History (Plotly)
- Small line chart, last 100 inferences
- Horizontal dashed threshold line
- Spikes colored red

### 7. Footer
- Stats bar: inference time, model size, accuracy, FFT size

## Pipeline (per inference cycle)
1. Generate signal from controls
2. Compute FFT (rfft, 128→64 bins)
3. Normalize with StandardScaler (from scaler.pkl)
4. Run TFLite interpreter (loaded from anomaly_model.tflite)
5. Compute MSE between input and output
6. Compare to threshold → anomaly yes/no
7. Update all visualizations

Target: < 50ms per cycle on modern hardware (Streamlit's caching + Plotly fast)

## Styling Directives (from high-end-design skill)
- **Background:** `#0A0A0A`
- **Cards:** `#141414` with `1px solid rgba(255,255,255,0.06)` border, `border-radius: 16px`, subtle `box-shadow`
- **Font:** Geist (sans), JetBrains Mono (numbers)
- **Accent:** `#00FF88` (green/normal), `#FF3366` (red/anomaly)
- **Transitions:** Custom cubic-bezier `(0.32, 0.72, 0, 1)` for all state changes
- **No Streamlit default chrome:** Hide hamburger menu, footer "Made with Streamlit", deploy button, default padding
- **Section padding:** `py-16` equivalent (CSS padding on containers)
- **Alert glow:** CSS `box-shadow` animation for LED pulse on anomaly

## Files
```
python/
  streamlit_app.py        ← NEW: main app
  train_anomaly_model.py  (existing)
  convert_model_to_header.py (existing)
  test_pipeline.py        (existing)
  requirements.txt        (updated with streamlit + plotly)
```

## Deployment
Single-command: `streamlit run streamlit_app.py`
Streamlit Cloud: connect GitHub repo, set entry point to `python/streamlit_app.py`
