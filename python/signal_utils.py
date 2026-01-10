import numpy as np
from config import SIGNAL_LENGTH, SAMPLE_RATE, FFT_SIZE

def generate_normal_signal(n_samples):
    signals = []
    for _ in range(n_samples):
        t = np.linspace(0, SIGNAL_LENGTH/SAMPLE_RATE, SIGNAL_LENGTH)
        signal = (np.sin(2*np.pi*10*t) + 
                 0.5*np.sin(2*np.pi*25*t) +
                 0.3*np.sin(2*np.pi*50*t) +
                 0.1*np.random.randn(SIGNAL_LENGTH))
        signals.append(signal)
    return np.array(signals)

def generate_anomaly_signal(n_samples):
    signals = []
    for _ in range(n_samples):
        t = np.linspace(0, SIGNAL_LENGTH/SAMPLE_RATE, SIGNAL_LENGTH)
        signal = (np.sin(2*np.pi*10*t) + 
                 0.5*np.sin(2*np.pi*25*t) +
                 1.5*np.sin(2*np.pi*150*t) +
                 0.8*np.sin(2*np.pi*200*t) +
                 0.3*np.random.randn(SIGNAL_LENGTH))
        signals.append(signal)
    return np.array(signals)

def compute_fft_features(signals):
    features = []
    for signal in signals:
        fft = np.fft.rfft(signal, n=SIGNAL_LENGTH)
        magnitude = np.abs(fft)[:FFT_SIZE]
        magnitude = magnitude / np.max(magnitude + 1e-10)
        features.append(magnitude)
    return np.array(features, dtype=np.float32)
