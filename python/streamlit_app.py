import sys, os, json, time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from config import SIGNAL_LENGTH, SAMPLE_RATE, FFT_SIZE
from numpy_model import predict as np_predict, mse as np_mse

st.set_page_config(page_title="Edge PDM", page_icon="⚙️", layout="wide")

@st.cache_resource
def load_params():
    with open("model_params.json") as f:
        p = json.load(f)
    return (
        np.array(p["scaler_mean"], dtype=np.float64),
        np.array(p["scaler_scale"], dtype=np.float64),
        p["threshold"]
    )

scaler_mean, scaler_scale, threshold = load_params()

def run_pipeline(signal):
    fft = np.fft.rfft(signal, n=SIGNAL_LENGTH)
    magnitude = np.abs(fft)[:FFT_SIZE]
    magnitude = magnitude / (np.max(magnitude) + 1e-10)
    x = (magnitude - scaler_mean) / scaler_scale
    reconstructed = np_predict(x)
    mse = np_mse(x, reconstructed)
    return magnitude, mse, mse > threshold

CUSTOM_CSS = """
<style>
    @import url('https://fonts.cdnfonts.com/css/geist');

    * { font-family: 'Geist', sans-serif; }

    .stApp { background: #0A0A0A; }

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
        transition: all 0.4s cubic-bezier(0.32,0.72,0,1);
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

    .stSlider label {
        color: rgba(255,255,255,0.6) !important;
        font-size: 0.8rem !important;
    }

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

def make_time_plot(signal, color="#00FF88"):
    t = np.linspace(0, SIGNAL_LENGTH / SAMPLE_RATE, SIGNAL_LENGTH)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal,
        line=dict(color=color, width=1.5),
        showlegend=False
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=180,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )
    return fig

def make_fft_plot(magnitude, is_anomaly):
    freqs = np.linspace(0, SAMPLE_RATE / 2, len(magnitude))
    color = "#FF3366" if is_anomaly else "#00FF88"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=freqs, y=magnitude,
        marker_color=color,
        marker_line_width=0,
        showlegend=False,
        opacity=0.85,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=180,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        bargap=0.05,
    )
    return fig

def make_error_history(errors, threshold):
    fig = go.Figure()
    colors = ["#FF3366" if e > threshold else "rgba(255,255,255,0.3)" for e in errors]
    fig.add_trace(go.Scatter(
        y=errors,
        mode="lines+markers",
        marker=dict(color=colors, size=3),
        line=dict(color="rgba(255,255,255,0.15)", width=1),
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

def generate_signal(is_anomaly, f1, f2, f3, f_high1, f_high2, noise):
    t = np.linspace(0, SIGNAL_LENGTH / SAMPLE_RATE, SIGNAL_LENGTH)
    signal = (f1 * np.sin(2 * np.pi * 10 * t) +
              f2 * np.sin(2 * np.pi * 25 * t) +
              f3 * np.sin(2 * np.pi * 50 * t))
    if is_anomaly:
        signal += f_high1 * np.sin(2 * np.pi * 150 * t)
        signal += f_high2 * np.sin(2 * np.pi * 200 * t)
    signal += noise * np.random.randn(SIGNAL_LENGTH)
    return signal

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown('<h1>Edge PDM</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predictive Maintenance System &middot; TinyML on ESP32</p>', unsafe_allow_html=True)

    if "error_history" not in st.session_state:
        st.session_state.error_history = []
    if "count_total" not in st.session_state:
        st.session_state.count_total = 0
    if "count_anomaly" not in st.session_state:
        st.session_state.count_anomaly = 0

    col_left, col_mid, col_right = st.columns([1.5, 3, 2])

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

        st.markdown('<div class="card" style="margin-top: 0.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Statistics</div>', unsafe_allow_html=True)
        total = st.session_state.count_total
        anomalies = st.session_state.count_anomaly
        rate = anomalies * 100 / max(total, 1)
        st.markdown(
            f'<div class="stat-grid">'
            f'  <div class="stat-item"><label>Inferences</label><span>{total}</span></div>'
            f'  <div class="stat-item"><label>Anomalies</label><span>{anomalies}</span></div>'
            f'  <div class="stat-item"><label>Rate</label><span>{rate:.0f}%</span></div>'
            f'  <div class="stat-item"><label>Model</label><span>24.9 KB</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_mid:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Time Domain</div>', unsafe_allow_html=True)
        signal = generate_signal(is_anomaly, f1, f2, f3, f_high1, f_high2, noise)
        wave_color = "#FF3366" if is_anomaly else "#00FF88"
        st.plotly_chart(make_time_plot(signal, wave_color), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top: 0.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">FFT Spectrum</div>', unsafe_allow_html=True)
        magnitude, mse, detected = run_pipeline(signal)
        st.plotly_chart(make_fft_plot(magnitude, detected), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

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

        if st.button("Reset History", width='stretch'):
            st.session_state.error_history = []
            st.session_state.count_total = 0
            st.session_state.count_anomaly = 0

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top: 0.75rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Error History (last 100)</div>', unsafe_allow_html=True)

        errors = st.session_state.error_history[-100:]
        if len(errors) > 1:
            st.plotly_chart(make_error_history(errors, threshold), width='stretch')
        else:
            st.markdown(
                '<p style="color:rgba(255,255,255,0.2);font-size:0.8rem;text-align:center;padding:1rem 0;">'
                'Waiting for data...</p>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.error_history.append(mse)
    st.session_state.count_total += 1
    if detected:
        st.session_state.count_anomaly += 1

    st.markdown(
        f'<div class="footer-bar">'
        f'  <span>Model: 24.9 KB</span>'
        f'  <span>Accuracy: 95%</span>'
        f'  <span>FFT: {FFT_SIZE} bins</span>'
        f'  <span>Threshold: {threshold:.2f}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    time.sleep(0.3)
    st.rerun()

if __name__ == "__main__":
    main()
