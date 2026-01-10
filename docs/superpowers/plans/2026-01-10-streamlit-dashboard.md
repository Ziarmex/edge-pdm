# Edge PDM Streamlit Dashboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a single-file Streamlit dashboard that visually demonstrates the Edge PDM anomaly detection pipeline (signal → FFT → TFLite → alert) in real-time.

**Architecture:** One Python file (`python/streamlit_app.py`) containing the full app. Custom CSS overrides Streamlit's default chrome for a premium dark theme. Plotly handles all charts. TFLite interpreter and scaler are loaded once via `@st.cache_resource`.

**Tech Stack:** Streamlit, Plotly, TensorFlow Lite, NumPy, scikit-learn

---

### Task 1: Update dependencies

**Files:**
- Modify: `python/requirements.txt`

- [ ] **Step 1: Add streamlit and plotly**

```
streamlit>=1.35.0
plotly>=5.20.0
```

Edit `python/requirements.txt` to append the two lines after the existing entries.

- [ ] **Step 2: Verify install**

Run: `pip install streamlit plotly`
Expected: No errors.

- [ ] **Step 3: Commit** (skip if user hasn't asked for commits)

---

### Task 2: Create the Streamlit app

**Files:**
- Create: `python/streamlit_app.py`

This task builds the entire app in one file (~250 lines). All code blocks below are sequential sections of the same file.

- [ ] **Step 1: Write imports, page config, and model loading**

Add to `python/streamlit_app.py`:

```python
import sys, os, json, pickle, time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from train_anomaly_model import generate_normal_signal, generate_anomaly_signal, compute_fft_features

st.set_page_config(page_title="Edge PDM", page_icon="⚙️", layout="wide")

@st.cache_resource
def load_model():
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path="anomaly_model.tflite")
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_params():
    with open("model_params.json") as f:
        return json.load(f)

interpreter, input_details, output_details = load_model()
scaler = load_scaler()
params = load_params()
threshold = params["threshold"]
```

- [ ] **Step 2: Write pipeline function**

Add after model loading:

```python
SIGNAL_LENGTH = 128
FFT_SIZE = 64

def run_pipeline(signal):
    fft = np.fft.rfft(signal, n=SIGNAL_LENGTH)
    magnitude = np.abs(fft)[:FFT_SIZE]
    magnitude = magnitude / (np.max(magnitude) + 1e-10)
    normalized = (magnitude - scaler.mean_) / scaler.scale_
    input_data = normalized.astype(np.float32).reshape(1, -1)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    mse = float(np.mean(np.square(input_data - output_data)))
    return magnitude, mse, mse > threshold
```

- [ ] **Step 3: Write custom CSS for premium theme**

Add after pipeline function:

```python
CUSTOM_CSS = """
<style>
    @import url('https://fonts.cdnfonts.com/css/geist');
    
    * { font-family: 'Geist', sans-serif; }
    
    .stApp {
        background: #0A0A0A;
    }
    
    /* Hide Streamlit chrome */
    #MainMenu, header, footer, .stDeployButton, .stAppDeployButton {
        display: none !important;
    }
    
    .block-container {
        padding: 1.5rem 2rem !important;
        max-width: 100% !important;
    }
    
    h1 {
        font-size: 1.75rem !important;
        font-weight: 400 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0 !important;
    }
    
    .subtitle {
        color: rgba(255,255,255,0.4);
        font-size: 0.85rem;
        margin-top: -0.25rem;
        margin-bottom: 1.5rem;
    }
    
    .card {
        background: #141414;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.25rem;
        height: 100%;
    }
    
    .card-label {
        color: rgba(255,255,255,0.35);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 500;
        margin-bottom: 0.75rem;
    }
    
    .led {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    .led-green {
        background: #00FF88;
        box-shadow: 0 0 12px rgba(0,255,136,0.5);
    }
    
    .led-red {
        background: #FF3366;
        box-shadow: 0 0 12px rgba(255,51,102,0.5);
        animation: pulse 1s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.15); }
    }
    
    .error-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 500;
        margin: 0.25rem 0;
    }
    
    .error-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .error-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.4s cubic-bezier(0.32,0.72,0,1);
    }
    
    .error-bar-green { background: #00FF88; }
    .error-bar-red { background: #FF3366; }
    
    .threshold-label {
        color: rgba(255,255,255,0.25);
        font-size: 0.7rem;
        margin-top: 0.25rem;
    }
    
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
    }
    
    .stat-item label {
        color: rgba(255,255,255,0.3);
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .stat-item span {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        color: #FFFFFF;
        display: block;
        margin-top: 0.15rem;
    }
    
    /* Slider styling */
    .stSlider label {
        color: rgba(255,255,255,0.6) !important;
        font-size: 0.8rem !important;
    }
    
    /* Radio / segmented button styling */
    .stRadio label {
        color: rgba(255,255,255,0.7) !important;
    }
    
    .footer-bar {
        border-top: 1px solid rgba(255,255,255,0.05);
        padding-top: 0.75rem;
        margin-top: 1.5rem;
        display: flex;
        gap: 2rem;
        justify-content: center;
        color: rgba(255,255,255,0.25);
        font-size: 0.75rem;
    }
    
    .footer-bar span { font-family: 'JetBrains Mono', monospace; }
</style>
"""
```

- [ ] **Step 4: Write chart creation functions**

```python
def make_time_plot(signal, color="#00FF88"):
    t = np.linspace(0, SIGNAL_LENGTH / 1000, SIGNAL_LENGTH)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal,
        line=dict(color=color, width=1.5),
        showlegend=False
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=180,
        xaxis=dict(title=None, showgrid=False, zeroline=False, visible=False),
        yaxis=dict(title=None, showgrid=False, zeroline=False, visible=False),
    )
    return fig

def make_fft_plot(magnitude, is_anomaly):
    freqs = np.linspace(0, 500, len(magnitude))
    colors = ["#00FF88" if not is_anomaly else "#FF3366"] * len(magnitude)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=freqs, y=magnitude,
        marker_color=colors,
        marker_line_width=0,
        showlegend=False
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=180,
        xaxis=dict(title=None, showgrid=False, zeroline=False, visible=False),
        yaxis=dict(title=None, showgrid=False, zeroline=False, visible=False),
        bargap=0.05,
    )
    return fig

def make_error_history(errors, threshold):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=errors, line=dict(color="rgba(255,255,255,0.3)", width=1),
        showlegend=False
    ))
    fig.add_hline(
        y=threshold, line_dash="dash",
        line_color="rgba(255,51,102,0.5)", line_width=1
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=120,
        xaxis=dict(visible=False, showgrid=False),
        yaxis=dict(visible=False, showgrid=False),
    )
    return fig
```

- [ ] **Step 5: Write UI layout and main loop**

```python
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1>Edge PDM</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predictive Maintenance System · TinyML on ESP32</p>', unsafe_allow_html=True)
    
    # Session state
    if "error_history" not in st.session_state:
        st.session_state.error_history = []
    if "count_total" not in st.session_state:
        st.session_state.count_total = 0
    if "count_anomaly" not in st.session_state:
        st.session_state.count_anomaly = 0
    
    col_left, col_mid, col_right = st.columns([1.5, 3, 2])
    
    # ── LEFT: Controls ──
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Signal Generator</div>', unsafe_allow_html=True)
        
        signal_type = st.radio("Type", ["Normal", "Anomaly"], horizontal=True, label_visibility="collapsed")
        is_anomaly = signal_type == "Anomaly"
        
        f1 = st.slider("10 Hz", 0.0, 2.0, 1.0, 0.1)
        f2 = st.slider("25 Hz", 0.0, 2.0, 0.5, 0.1)
        f3 = st.slider("50 Hz", 0.0, 2.0, 0.3, 0.1)
        f_high1 = st.slider("150 Hz (anomaly)", 0.0, 3.0, 1.5 if is_anomaly else 0.0, 0.1)
        f_high2 = st.slider("200 Hz (anomaly)", 0.0, 3.0, 0.8 if is_anomaly else 0.0, 0.1)
        noise = st.slider("Noise", 0.0, 0.5, 0.1, 0.05)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Stats card
        st.markdown('<div class="card" style="margin-top: 0.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Statistics</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="stat-grid">'
            f'  <div class="stat-item"><label>Inferences</label><span>{st.session_state.count_total}</span></div>'
            f'  <div class="stat-item"><label>Anomalies</label><span>{st.session_state.count_anomaly}</span></div>'
            f'  <div class="stat-item"><label>Rate</label><span>'
            f'    {st.session_state.count_anomaly * 100 / max(st.session_state.count_total, 1):.0f}%'
            f'  </span></div>'
            f'  <div class="stat-item"><label>Model</label><span>24.9 KB</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ── MID: Visualizations ──
    with col_mid:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Time Domain</div>', unsafe_allow_html=True)
        signal = generate_signal(is_anomaly, f1, f2, f3, f_high1, f_high2, noise)
        st.plotly_chart(make_time_plot(signal, "#FF3366" if is_anomaly else "#00FF88"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card" style="margin-top: 0.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">FFT Spectrum</div>', unsafe_allow_html=True)
        magnitude, mse, detected = run_pipeline(signal)
        st.plotly_chart(make_fft_plot(magnitude, detected), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ── RIGHT: Status ──
    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Detection Status</div>', unsafe_allow_html=True)
        
        led_class = "led-red" if detected else "led-green"
        led_text = "ANOMALY DETECTED" if detected else "NORMAL"
        led_color = "#FF3366" if detected else "#00FF88"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">'
            f'  <span class="led {led_class}"></span>'
            f'  <span style="color:{led_color};font-size:1rem;font-weight:500;">{led_text}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.markdown('<div style="margin: 1rem 0;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Reconstruction Error</div>', unsafe_allow_html=True)
        error_pct = min(mse / threshold, 1.5)
        bar_class = "error-bar-red" if detected else "error-bar-green"
        st.markdown(
            f'<div class="error-value" style="color:{led_color};">{mse:.4f}</div>'
            f'<div class="error-bar-bg">'
            f'  <div class="error-bar-fill {bar_class}" style="width:{min(error_pct * 100, 100):.0f}%"></div>'
            f'</div>'
            f'<div class="threshold-label">Threshold: {threshold:.4f}</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Reset History", use_container_width=True):
            st.session_state.error_history = []
            st.session_state.count_total = 0
            st.session_state.count_anomaly = 0
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Error history card
        st.markdown('<div class="card" style="margin-top: 0.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Error History (last 100)</div>', unsafe_allow_html=True)
        
        errors = st.session_state.error_history[-100:]
        if len(errors) > 1:
            st.plotly_chart(make_error_history(errors, threshold), use_container_width=True)
        else:
            st.markdown(
                '<p style="color:rgba(255,255,255,0.2);font-size:0.8rem;text-align:center;padding:1rem 0;">'
                'Waiting for data...</p>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Update history ──
    st.session_state.error_history.append(mse)
    st.session_state.count_total += 1
    if detected:
        st.session_state.count_anomaly += 1
    
    # Footer
    st.markdown(
        f'<div class="footer-bar">'
        f'  <span>Model: 24.9 KB</span>'
        f'  <span>Accuracy: 95%</span>'
        f'  <span>FFT: {FFT_SIZE} bins</span>'
        f'  <span>Threshold: {threshold:.2f}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Auto-refresh
    time.sleep(0.3)
    st.rerun()

def generate_signal(is_anomaly, f1, f2, f3, f_high1, f_high2, noise):
    t = np.linspace(0, SIGNAL_LENGTH / 1000, SIGNAL_LENGTH)
    signal = (f1 * np.sin(2 * np.pi * 10 * t) +
              f2 * np.sin(2 * np.pi * 25 * t) +
              f3 * np.sin(2 * np.pi * 50 * t))
    if is_anomaly:
        signal += f_high1 * np.sin(2 * np.pi * 150 * t)
        signal += f_high2 * np.sin(2 * np.pi * 200 * t)
    signal += noise * np.random.randn(SIGNAL_LENGTH)
    return signal

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify the app starts**

```bash
cd /path/to/edge-pdm-main/python
streamlit run streamlit_app.py
```

Expected: App opens in browser at `http://localhost:8501` showing the dashboard with a dark theme, controls, signal plots, and live detection.

---

## Self-Review

**Spec coverage:** The spec calls for 7 components (controls, time plot, FFT plot, LED, error gauge, stats, error history) and all 7 are implemented. The premium dark theme CSS, glass cards, LED pulse animation, and error bar transitions are all present.

**Placeholder scan:** No "TBD", "TODO", or incomplete code. Every step has the actual implementation.

**Type consistency:** `generate_signal()` signature matches the call site. `run_pipeline()` returns `(magnitude, mse, detected)` consistent with all usage. Interpreter/scaler loading uses `@st.cache_resource` for efficiency as specified.
