/*
 * Système de Maintenance Prédictive Embarquée (TinyML)
 * ESP32 avec TensorFlow Lite et FFT
 * 
 * Fonctionnalités :
 * - Acquisition de signaux (simulés ou ADC)
 * - Analyse FFT en temps réel
 * - Détection d'anomalies via TFLite
 * - Alertes LED
 */

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "arduinoFFT.h"

// ============ Configuration ============
#define SAMPLE_RATE 1000        // Hz
#define SIGNAL_LENGTH 128       // Échantillons
#define FFT_SIZE 64             // Moitié du signal
#define LED_PIN 2               // LED built-in ESP32

// Simulation ou ADC réel
#define USE_SIMULATION true     // true pour simulation, false pour ADC
#define ADC_PIN 34              // Pin ADC si USE_SIMULATION = false

// ============ Variables globales ============
arduinoFFT FFT = arduinoFFT();

// Buffers pour FFT
double vReal[SIGNAL_LENGTH];
double vImag[SIGNAL_LENGTH];
float fftFeatures[FFT_SIZE];

// TensorFlow Lite
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Mémoire pour TFLite (à ajuster si nécessaire)
constexpr int kTensorArenaSize = 30000;
uint8_t tensor_arena[kTensorArenaSize];

// Statistiques
unsigned long inferenceCount = 0;
unsigned long anomalyCount = 0;
unsigned long totalInferenceTime = 0;

// ============ Modèle et paramètres TFLite ============
#include "model_data.h"     // model_data[] + model_data_len
#include "scaler_params.h"  // scaler_mean[], scaler_scale[], ANOMALY_THRESHOLD, FFT_SIZE

// ============ Fonctions ============

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  
  Serial.println("\n=== Edge Predictive Maintenance System ===");
  Serial.println("TinyML sur ESP32");
  
  // Configuration LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // Configuration ADC si nécessaire
  if (!USE_SIMULATION) {
    analogReadResolution(12);  // 12-bit ADC
    pinMode(ADC_PIN, INPUT);
  }
  
  // Chargement du modèle TFLite
  Serial.println("Chargement du modèle TFLite...");
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Erreur: Version du modèle %d != %d\n", 
                  model->version(), TFLITE_SCHEMA_VERSION);
    while(1);
  }
  
  // Opérations
  static tflite::AllOpsResolver resolver;
  
  // Interpréteur
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  
  // Allocation des tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Erreur: Allocation des tensors échouée!");
    while(1);
  }
  
  // Obtention des tensors I/O
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.printf("Modèle chargé. Input: [%d], Output: [%d]\n", 
                input->dims->data[1], output->dims->data[1]);
  Serial.printf("Mémoire utilisée: %d/%d bytes\n", 
                interpreter->arena_used_bytes(), kTensorArenaSize);
  
  Serial.println("\nSystème prêt. Démarrage de la détection...\n");
  delay(1000);
}

void loop() {
  unsigned long startTime = millis();
  
  // 1. Acquisition du signal
  acquireSignal();
  
  // 2. Calcul FFT
  computeFFT();
  
  // 3. Normalisation des features
  normalizeFeatures();
  
  // 4. Copie dans le tensor d'entrée
  for (int i = 0; i < FFT_SIZE; i++) {
    input->data.f[i] = fftFeatures[i];
  }
  
  // 5. Inférence
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Erreur: Inférence échouée!");
    return;
  }
  
  // 6. Calcul de l'erreur de reconstruction (MSE)
  float reconstructionError = 0.0;
  for (int i = 0; i < FFT_SIZE; i++) {
    float diff = input->data.f[i] - output->data.f[i];
    reconstructionError += diff * diff;
  }
  reconstructionError /= FFT_SIZE;
  
  // 7. Détection d'anomalie
  bool isAnomaly = reconstructionError > ANOMALY_THRESHOLD;
  
  // 8. Alerte
  if (isAnomaly) {
    digitalWrite(LED_PIN, HIGH);
    anomalyCount++;
    Serial.println(" ANOMALIE DÉTECTÉE!");
  } else {
    digitalWrite(LED_PIN, LOW);
  }
  
  // 9. Statistiques
  inferenceCount++;
  unsigned long inferenceTime = millis() - startTime;
  totalInferenceTime += inferenceTime;
  
  // Affichage périodique
  if (inferenceCount % 10 == 0) {
    float avgTime = totalInferenceTime / (float)inferenceCount;
    Serial.printf("Inférence #%lu | Error: %.6f | Anomalies: %lu/%lu | Temps: %lums (Moy: %.1fms)\n",
                  inferenceCount, reconstructionError, anomalyCount, 
                  inferenceCount, inferenceTime, avgTime);
  }
  
  // Délai entre acquisitions
  delay(500);
}

// ============ Acquisition du signal ============
void acquireSignal() {
  if (USE_SIMULATION) {
    // Signal simulé (à adapter selon le besoin)
    float t;
    bool simulateAnomaly = (random(100) < 10);  // 10% de chance d'anomalie
    
    for (int i = 0; i < SIGNAL_LENGTH; i++) {
      t = i / (float)SAMPLE_RATE;
      
      if (simulateAnomaly) {
        // Signal anormal (hautes fréquences)
        vReal[i] = sin(2 * PI * 10 * t) + 
                   0.5 * sin(2 * PI * 25 * t) +
                   1.5 * sin(2 * PI * 150 * t) +
                   0.8 * sin(2 * PI * 200 * t);
      } else {
        // Signal normal
        vReal[i] = sin(2 * PI * 10 * t) + 
                   0.5 * sin(2 * PI * 25 * t) +
                   0.3 * sin(2 * PI * 50 * t);
      }
      
      // Bruit
      vReal[i] += (random(-100, 100) / 1000.0);
      vImag[i] = 0.0;
    }
  } else {
    // Acquisition ADC réelle
    for (int i = 0; i < SIGNAL_LENGTH; i++) {
      vReal[i] = analogRead(ADC_PIN);
      vImag[i] = 0.0;
      delayMicroseconds(1000000 / SAMPLE_RATE);  // Respect du sample rate
    }
  }
}

// ============ Calcul FFT ============
void computeFFT() {
  // FFT
  FFT.Windowing(vReal, SIGNAL_LENGTH, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.Compute(vReal, vImag, SIGNAL_LENGTH, FFT_FORWARD);
  FFT.ComplexToMagnitude(vReal, vImag, SIGNAL_LENGTH);
  
  // Extraction des FFT_SIZE premières valeurs (moitié du spectre)
  float maxMagnitude = 0.0;
  for (int i = 0; i < FFT_SIZE; i++) {
    fftFeatures[i] = vReal[i];
    if (fftFeatures[i] > maxMagnitude) {
      maxMagnitude = fftFeatures[i];
    }
  }
  
  // Normalisation par la magnitude max
  if (maxMagnitude > 0.0) {
    for (int i = 0; i < FFT_SIZE; i++) {
      fftFeatures[i] /= maxMagnitude;
    }
  }
}

// ============ Normalisation des features ============
void normalizeFeatures() {
  // StandardScaler: (x - mean) / scale
  for (int i = 0; i < FFT_SIZE; i++) {
    fftFeatures[i] = (fftFeatures[i] - scaler_mean[i]) / scaler_scale[i];
  }
}


