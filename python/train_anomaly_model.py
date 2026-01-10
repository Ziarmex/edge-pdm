"""
Entraînement du modèle d'autoencoder pour détection d'anomalies
Génère un modèle TFLite optimisé pour ESP32
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

from config import SAMPLE_RATE, SIGNAL_LENGTH, FFT_SIZE, N_NORMAL_SAMPLES, N_ANOMALY_SAMPLES

def generate_normal_signal(n_samples):
    """Génère des signaux normaux (vibrations saines)"""
    signals = []
    for _ in range(n_samples):
        t = np.linspace(0, SIGNAL_LENGTH/SAMPLE_RATE, SIGNAL_LENGTH)
        # Signal normal : quelques harmoniques basses fréquences
        signal = (np.sin(2*np.pi*10*t) + 
                 0.5*np.sin(2*np.pi*25*t) +
                 0.3*np.sin(2*np.pi*50*t) +
                 0.1*np.random.randn(SIGNAL_LENGTH))  # Bruit
        signals.append(signal)
    return np.array(signals)

def generate_anomaly_signal(n_samples):
    """Génère des signaux anormaux (vibrations défectueuses)"""
    signals = []
    for _ in range(n_samples):
        t = np.linspace(0, SIGNAL_LENGTH/SAMPLE_RATE, SIGNAL_LENGTH)
        # Signal anormal : harmoniques hautes fréquences, amplitude élevée
        signal = (np.sin(2*np.pi*10*t) + 
                 0.5*np.sin(2*np.pi*25*t) +
                 1.5*np.sin(2*np.pi*150*t) +  # Haute fréquence anormale
                 0.8*np.sin(2*np.pi*200*t) +
                 0.3*np.random.randn(SIGNAL_LENGTH))
        signals.append(signal)
    return np.array(signals)

def compute_fft_features(signals):
    """Calcule les features FFT pour chaque signal"""
    features = []
    for signal in signals:
        # FFT et magnitude
        fft = np.fft.rfft(signal, n=SIGNAL_LENGTH)
        magnitude = np.abs(fft)[:FFT_SIZE]
        # Normalisation
        magnitude = magnitude / np.max(magnitude + 1e-10)
        features.append(magnitude)
    return np.array(features, dtype=np.float32)

def create_autoencoder(input_dim):
    """Crée un autoencoder léger pour ESP32"""
    # Encoder
    encoder_input = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(32, activation='relu')(encoder_input)
    x = keras.layers.Dense(16, activation='relu')(x)
    encoded = keras.layers.Dense(8, activation='relu')(x)
    
    # Decoder
    x = keras.layers.Dense(16, activation='relu')(encoded)
    x = keras.layers.Dense(32, activation='relu')(x)
    decoded = keras.layers.Dense(input_dim, activation='linear')(x)
    
    # Modèle complet
    autoencoder = keras.Model(encoder_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return autoencoder

def main():
    print("=== Génération des données ===")
    # Génération des signaux
    normal_signals = generate_normal_signal(N_NORMAL_SAMPLES)
    anomaly_signals = generate_anomaly_signal(N_ANOMALY_SAMPLES)
    
    # Extraction des features FFT
    print("Extraction des features FFT...")
    normal_features = compute_fft_features(normal_signals)
    anomaly_features = compute_fft_features(anomaly_signals)
    
    # Split train/test
    split_idx = int(0.8 * len(normal_features))
    X_train = normal_features[:split_idx]
    X_test = normal_features[split_idx:]
    X_test_anomaly = anomaly_features
    
    print(f"Train: {X_train.shape}, Test normal: {X_test.shape}, Test anomaly: {X_test_anomaly.shape}")
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_test_anomaly_scaled = scaler.transform(X_test_anomaly)
    
    # Sauvegarde du scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler sauvegardé : scaler.pkl")
    
    print("\n=== Entraînement du modèle ===")
    # Création et entraînement
    model = create_autoencoder(FFT_SIZE)
    model.summary()
    
    history = model.fit(
        X_train_scaled, X_train_scaled,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Évaluation
    print("\n=== Évaluation ===")
    # Reconstruction errors
    train_pred = model.predict(X_train_scaled)
    train_mse = np.mean(np.square(X_train_scaled - train_pred), axis=1)
    
    test_pred = model.predict(X_test_scaled)
    test_mse = np.mean(np.square(X_test_scaled - test_pred), axis=1)
    
    test_anomaly_pred = model.predict(X_test_anomaly_scaled)
    test_anomaly_mse = np.mean(np.square(X_test_anomaly_scaled - test_anomaly_pred), axis=1)
    
    # Seuil de détection (95e percentile des erreurs normales)
    threshold = np.percentile(train_mse, 95)
    print(f"Seuil de détection : {threshold:.6f}")
    
    # Accuracy
    normal_correct = np.sum(test_mse < threshold)
    anomaly_correct = np.sum(test_anomaly_mse >= threshold)
    total_correct = normal_correct + anomaly_correct
    total_samples = len(test_mse) + len(test_anomaly_mse)
    accuracy = total_correct / total_samples
    
    print(f"Précision normale : {normal_correct}/{len(test_mse)} = {normal_correct/len(test_mse)*100:.1f}%")
    print(f"Détection anomalie : {anomaly_correct}/{len(test_anomaly_mse)} = {anomaly_correct/len(test_anomaly_mse)*100:.1f}%")
    print(f"Accuracy totale : {accuracy*100:.1f}%")
    
    # Visualisation
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    plt.subplot(1, 2, 2)
    plt.hist(test_mse, bins=50, alpha=0.5, label='Normal')
    plt.hist(test_anomaly_mse, bins=50, alpha=0.5, label='Anomalie')
    plt.axvline(threshold, color='r', linestyle='--', label='Seuil')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.title('Distribution des erreurs')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    print("Graphiques sauvegardés : training_results.png")
    
    # Conversion en TFLite
    print("\n=== Conversion TFLite ===")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    
    tflite_model = converter.convert()
    
    # Sauvegarde
    with open('anomaly_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    model_size = len(tflite_model) / 1024
    print(f"Modèle TFLite sauvegardé : anomaly_model.tflite ({model_size:.1f} KB)")
    
    # Sauvegarde des paramètres
    params = {
        'threshold': float(threshold),
        'fft_size': FFT_SIZE,
        'signal_length': SIGNAL_LENGTH,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }
    
    import json
    with open('model_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    print("Paramètres sauvegardés : model_params.json")
    
    print("\n=== Entraînement terminé ===")
    print(f"Fichiers générés :")
    print(f"  - anomaly_model.tflite ({model_size:.1f} KB)")
    print(f"  - model_params.json")
    print(f"  - scaler.pkl")
    print(f"  - training_results.png")

if __name__ == "__main__":
    main()
