"""
Convertit le modèle TFLite en fichier header C pour ESP32
Génère également le code pour les scalers
"""

import json
import sys

def tflite_to_header(tflite_file, output_file):
    """Convertit un fichier TFLite en header C"""
    print(f"Lecture de {tflite_file}...")
    
    with open(tflite_file, 'rb') as f:
        model_data = f.read()
    
    model_size = len(model_data)
    print(f"Taille du modèle: {model_size} bytes ({model_size/1024:.2f} KB)")
    
    # Génération du fichier header
    with open(output_file, 'w') as f:
        f.write("/*\n")
        f.write(" * Modèle TFLite converti automatiquement\n")
        f.write(f" * Taille: {model_size} bytes\n")
        f.write(" */\n\n")
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        
        # Données du modèle
        f.write("alignas(8) const unsigned char model_data[] = {\n")
        
        # Écriture en hexadécimal, 12 bytes par ligne
        for i in range(0, model_size, 12):
            line = "  "
            chunk = model_data[i:min(i+12, model_size)]
            line += ", ".join(f"0x{b:02x}" for b in chunk)
            if i + 12 < model_size:
                line += ","
            f.write(line + "\n")
        
        f.write("};\n")
        f.write(f"const unsigned int model_data_len = {model_size};\n\n")
        f.write("#endif  // MODEL_DATA_H\n")
    
    print(f"Fichier header généré: {output_file}")
    return model_size

def generate_scaler_code(params_file, output_file):
    """Génère le code C pour les scalers"""
    print(f"\nLecture de {params_file}...")
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    fft_size = params['fft_size']
    threshold = params['threshold']
    scaler_mean = params['scaler_mean']
    scaler_scale = params['scaler_scale']
    
    print(f"FFT Size: {fft_size}")
    print(f"Threshold: {threshold}")
    
    with open(output_file, 'w') as f:
        f.write("/*\n")
        f.write(" * Paramètres du scaler et seuil de détection\n")
        f.write(" * Généré automatiquement depuis model_params.json\n")
        f.write(" */\n\n")
        f.write("#ifndef SCALER_PARAMS_H\n")
        f.write("#define SCALER_PARAMS_H\n\n")
        
        # Seuil
        f.write(f"#define ANOMALY_THRESHOLD {threshold}f\n")
        f.write(f"#define FFT_SIZE {fft_size}\n\n")
        
        # Scaler mean
        f.write(f"const float scaler_mean[{fft_size}] = {{\n")
        for i in range(0, fft_size, 8):
            line = "  "
            chunk = scaler_mean[i:min(i+8, fft_size)]
            line += ", ".join(f"{v:.8f}f" for v in chunk)
            if i + 8 < fft_size:
                line += ","
            f.write(line + "\n")
        f.write("};\n\n")
        
        # Scaler scale
        f.write(f"const float scaler_scale[{fft_size}] = {{\n")
        for i in range(0, fft_size, 8):
            line = "  "
            chunk = scaler_scale[i:min(i+8, fft_size)]
            line += ", ".join(f"{v:.8f}f" for v in chunk)
            if i + 8 < fft_size:
                line += ","
            f.write(line + "\n")
        f.write("};\n\n")
        
        # Fonction d'initialisation
        f.write("void initializeScalers() {\n")
        f.write("  // Scalers déjà définis comme constantes globales\n")
        f.write('  Serial.println("Scalers initialisés depuis les paramètres");\n')
        f.write("}\n\n")
        
        f.write("#endif  // SCALER_PARAMS_H\n")
    
    print(f"Fichier scaler généré: {output_file}")

def generate_readme(model_size):
    """Génère un README avec les instructions"""
    readme_content = f"""# Installation du Modèle sur ESP32

## Fichiers générés

1. **model_data.h** ({model_size/1024:.2f} KB)
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
   void initializeScalers() {{
     // Valeurs par défaut
     for (int i = 0; i < FFT_SIZE; i++) {{
       scaler_mean[i] = 0.0;
       scaler_scale[i] = 1.0;
     }}
   }}
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
- **Mémoire utilisée**: ~{model_size/1024:.0f} KB (modèle) + ~30 KB (arena)
- **Accuracy**: > 80% (selon les tests Python)
- **Seuil actuel**: {0.05}

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
│   ├── Model Data (~{model_size/1024:.0f}KB)
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
"""
    
    with open('INSTALLATION.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("README généré: INSTALLATION.md")

def main():
    print("=== Conversion du modèle pour ESP32 ===\n")
    
    # Fichiers d'entrée
    tflite_file = 'anomaly_model.tflite'
    params_file = 'model_params.json'
    
    # Fichiers de sortie
    model_header = 'model_data.h'
    scaler_header = 'scaler_params.h'
    
    try:
        # Conversion du modèle
        model_size = tflite_to_header(tflite_file, model_header)
        
        # Génération des scalers
        generate_scaler_code(params_file, scaler_header)
        
        # Génération du README
        generate_readme(model_size)
        
        print("\n=== Conversion terminée avec succès ===")
        print("\nFichiers générés :")
        print(f"  ✓ {model_header}")
        print(f"  ✓ {scaler_header}")
        print(f"  ✓ INSTALLATION.md")
        print("\nSuivez les instructions dans INSTALLATION.md pour flasher l'ESP32")
        
    except FileNotFoundError as e:
        print(f"\n❌ Erreur: Fichier non trouvé - {e}")
        print("Assurez-vous d'avoir exécuté train_anomaly_model.py d'abord")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
