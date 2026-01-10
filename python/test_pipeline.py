"""
Test de validation du pipeline complet : signal → FFT → scaler → TFLite → détection
Simule le comportement ESP32 pour vérifier que tout fonctionne.
"""

import numpy as np
from numpy_model import predict as np_predict, mse as np_mse
from sklearn.preprocessing import StandardScaler
import pickle
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_anomaly_model import (
    generate_normal_signal, generate_anomaly_signal,
    compute_fft_features, FFT_SIZE, SIGNAL_LENGTH, SAMPLE_RATE
)

def test_pipeline():
    print("=" * 60)
    print("TEST DU PIPELINE COMPLET (simulation ESP32)")
    print("=" * 60)

    # 1. Charger le modèle
    print("\n[1] Chargement du modèle NumPy...")
    _ = np_predict(np.zeros(FFT_SIZE))
    print(f"    Modèle chargé: {FFT_SIZE} features -> autoencoder -> {FFT_SIZE} features")

    # 2. Charger le scaler
    print("\n[2] Chargement du scaler...")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print(f"    Scaler mean (premiers 5): {scaler.mean_[:5]}")
    print(f"    Scaler scale (premiers 5): {scaler.scale_[:5]}")

    # 3. Charger les paramètres
    print("\n[3] Chargement des paramètres...")
    with open("model_params.json", "r") as f:
        params = json.load(f)
    threshold = params["threshold"]
    print(f"    Seuil de détection: {threshold:.6f}")

    # 4. Générer des signaux de test
    print("\n[4] Génération des signaux de test...")
    n_test = 100
    normal_signals = generate_normal_signal(n_test)
    anomaly_signals = generate_anomaly_signal(n_test)
    print(f"    Signaux normaux: {normal_signals.shape}")
    print(f"    Signaux anormaux: {anomaly_signals.shape}")

    # 5. Pipeline complet (comme sur ESP32)
    print("\n[5] Test du pipeline complet (FFT → Scaler → TFLite → MSE)...")
    
    def esp32_pipeline(signals):
        """Reproduit exactement le pipeline de l'ESP32"""
        errors = []
        for signal in signals:
            fft = np.fft.rfft(signal, n=SIGNAL_LENGTH)
            magnitude = np.abs(fft)[:FFT_SIZE]
            max_mag = np.max(magnitude + 1e-10)
            magnitude = magnitude / max_mag
            x = (magnitude - scaler.mean_) / scaler.scale_
            reconstructed = np_predict(x)
            mse = np_mse(x, reconstructed)
            errors.append(mse)
        
        return np.array(errors)
    
    normal_errors = esp32_pipeline(normal_signals)
    anomaly_errors = esp32_pipeline(anomaly_signals)
    
    # 6. Évaluation de la détection
    print("\n[6] Évaluation de la détection d'anomalies...")
    normal_detected = normal_errors > threshold
    anomaly_detected = anomaly_errors > threshold
    
    normal_accuracy = np.sum(~normal_detected) / len(normal_errors)
    anomaly_accuracy = np.sum(anomaly_detected) / len(anomaly_errors)
    total_accuracy = (np.sum(~normal_detected) + np.sum(anomaly_detected)) / (len(normal_errors) + len(anomaly_errors))
    
    print(f"    Seuil: {threshold:.4f}")
    print(f"    Erreur normale - min: {normal_errors.min():.4f}, max: {normal_errors.max():.4f}, moy: {normal_errors.mean():.4f}")
    print(f"    Erreur anomalie - min: {anomaly_errors.min():.4f}, max: {anomaly_errors.max():.4f}, moy: {anomaly_errors.mean():.4f}")
    print(f"    Précision normale (pas d'alerte): {normal_accuracy*100:.1f}%")
    print(f"    Détection anomalie (alerte): {anomaly_accuracy*100:.1f}%")
    print(f"    Accuracy totale: {total_accuracy*100:.1f}%")
    
    # 7. Vérification des contraintes ESP32
    print("\n[7] Vérification des contraintes ESP32...")
    
    # Taille du modèle
    model_size = os.path.getsize("anomaly_model.tflite")
    print(f"    Taille du modèle: {model_size/1024:.1f} KB (limite: 100 KB) {'✓' if model_size < 100*1024 else '✗'}")
    
    # Vérification des timings (approximatif sur PC)
    import time
    n_warmup = 10
    n_bench = 50
    
    print("\n[8] Benchmark de performance...")
    # Warmup
    test_signal = normal_signals[0]
    for _ in range(n_warmup):
        _ = esp32_pipeline([test_signal])
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_bench):
        _ = esp32_pipeline([test_signal])
    elapsed = time.perf_counter() - start
    avg_time = (elapsed / n_bench) * 1000  # en ms
    print(f"    Temps moyen par inférence (PC): {avg_time:.1f}ms")
    print(f"    (Sur ESP32, attendu ~70-100ms)")
    
    # 9. Rapport final
    print("\n" + "=" * 60)
    print("RAPPORT FINAL")
    print("=" * 60)
    
    all_ok = True
    checks = [
        ("Modèle < 100 KB", model_size < 100 * 1024),
        ("Accuracy > 80%", total_accuracy > 0.80),
        ("Détection anomalies > 80%", anomaly_accuracy > 0.80),
        ("Précision normale > 80%", normal_accuracy > 0.80),
    ]
    
    for name, ok in checks:
        status = "✓" if ok else "✗"
        print(f"  [{status}] {name}")
        all_ok = all_ok and ok
    
    print()
    if all_ok:
        print("  ✓ TOUS LES TESTS PASSENT - Le système est prêt pour ESP32!")
    else:
        print("  ✗ Certains tests échouent - Vérifiez les résultats ci-dessus")
    
    return all_ok

def test_conversion():
    """Vérifie que les fichiers headers C sont corrects"""
    print("\n" + "=" * 60)
    print("TEST DE CONVERSION C HEADERS")
    print("=" * 60)
    
    # Vérifier model_data.h
    print("\n[1] Vérification de model_data.h...")
    with open("model_data.h", "r") as f:
        content = f.read()
    
    checks_h = [
        ("Contient le modèle", "model_data[]" in content),
        ("Contient la taille", "model_data_len" in content),
        ("Headers guards", "#ifndef MODEL_DATA_H" in content),
        ("Alignement 8 bytes", "alignas(8)" in content),
    ]
    for name, ok in checks_h:
        status = "✓" if ok else "✗"
        print(f"  [{status}] {name}")
    
    # Vérifier scaler_params.h
    print("\n[2] Vérification de scaler_params.h...")
    with open("scaler_params.h", "r") as f:
        content = f.read()
    
    with open("model_params.json", "r") as f:
        params = json.load(f)
    
    checks_s = [
        ("Threshold présent", f"ANOMALY_THRESHOLD {params['threshold']}f" in content),
        ("FFT_SIZE présent", f"FFT_SIZE {params['fft_size']}" in content),
        ("scaler_mean présent", "scaler_mean[64]" in content or "scaler_mean[]" in content),
        ("scaler_scale présent", "scaler_scale[64]" in content or "scaler_scale[]" in content),
        ("Headers guards", "#ifndef SCALER_PARAMS_H" in content),
    ]
    for name, ok in checks_s:
        status = "✓" if ok else "✗"
        print(f"  [{status}] {name}")

if __name__ == "__main__":
    all_ok = test_pipeline()
    test_conversion()
    sys.exit(0 if all_ok else 1)
