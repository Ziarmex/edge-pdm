# Installation du Modèle sur ESP32

## Fichiers générés

1. **model_data.h** (24.86 KB)
   - Contient le modèle TFLite converti en array C
   - À placer dans le dossier de votre sketch Arduino

2. **scaler_params.h**
   - Contient les paramètres de normalisation (mean/scale)
   - Contient le seuil de détection d'anomalies
   - À placer dans le dossier de votre sketch Arduino

## Instructions d'installation

### 1. Bibliothèques requises

Installez via le Library Manager Arduino IDE :

- **TensorFlowLite_ESP32** (par TensorFlow Authors)
- **arduinoFFT** (par Enrique Condes)

### 2. Configuration du projet

1. Créez un nouveau sketch dans Arduino IDE
2. Copiez le code de `edge_pdm_esp32.ino`
3. Placez `model_data.h` et `scaler_params.h` dans le même dossier
4. Dans le code, remplacez :
   ```cpp
   #include "model_data.h"
   ```
   et ajoutez :
   ```cpp
   #include "scaler_params.h"
   ```

5. Dans la fonction `initializeScalers()`, remplacez :
   ```cpp
   void initializeScalers() {
     // Valeurs par défaut
     for (int i = 0; i < FFT_SIZE; i++) {
       scaler_mean[i] = 0.0;
       scaler_scale[i] = 1.0;
     }
   }
   ```
   par :
   ```cpp
   // Supprimez cette fonction, elle est maintenant dans scaler_params.h
   ```

6. Supprimez les déclarations de `scaler_mean` et `scaler_scale` dans le .ino
   (elles sont maintenant dans scaler_params.h)

### 3. Compilation et upload

1. Sélectionnez votre board ESP32 (ex: ESP32 Dev Module)
2. Sélectionnez le port COM
3. Paramètres recommandés :
   - Flash Size: 4MB
   - Partition Scheme: Default 4MB with spiffs
   - CPU Frequency: 240MHz
4. Cliquez sur "Upload"

### 4. Test

1. Ouvrez le Serial Monitor (115200 baud)
2. Vous devriez voir :
   ```
   === Edge Predictive Maintenance System ===
   Chargement du modèle TFLite...
   Modèle chargé. Input: [64], Output: [64]
   Système prêt. Démarrage de la détection...
   ```
3. La LED built-in (GPIO2) s'allume lors d'une détection d'anomalie

### 5. Utilisation avec capteur réel

Pour utiliser un accéléromètre au lieu de signaux simulés :

1. Connectez votre capteur à l'ESP32 (ex: ADXL345 sur ADC pin 34)
2. Dans le code, changez :
   ```cpp
   #define USE_SIMULATION false
   #define ADC_PIN 34
   ```

## Performances attendues

- **Temps d'inférence**: < 100ms
- **Mémoire utilisée**: ~25 KB (modèle) + ~30 KB (arena)
- **Accuracy**: > 80% (selon les tests Python)
- **Seuil actuel**: 0.05

## Dépannage

### Erreur "Allocation des tensors échouée"
- Augmentez `kTensorArenaSize` dans le code (actuellement 30000)

### Erreur de compilation "model_data not found"
- Vérifiez que model_data.h est bien dans le dossier du sketch

### Inférence trop lente
- Vérifiez que CPU Frequency est à 240MHz
- Réduisez FFT_SIZE si nécessaire

### Fausses détections
- Ajustez ANOMALY_THRESHOLD dans scaler_params.h
- Ré-entraînez le modèle avec plus de données

## Architecture mémoire ESP32

```
Flash (4MB)
├── Bootloader
├── Partition Table
├── Application (~1.5MB)
│   ├── Code
│   ├── Model Data (~25KB)
│   └── Constantes
└── SPIFFS

RAM (520KB)
├── DRAM (~160KB libre)
│   ├── Variables globales
│   ├── Tensor Arena (30KB)
│   ├── FFT Buffers
│   └── Stack
└── IRAM (code exécutable)
```

## Contacts et support

- Documentation TensorFlow Lite: https://www.tensorflow.org/lite/microcontrollers
- ESP32 Forums: https://esp32.com
- ArduinoFFT: https://github.com/kosme/arduinoFFT
