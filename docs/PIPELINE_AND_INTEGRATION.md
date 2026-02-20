# MERS Pipeline & Integration Documentation

## 1. Training Pipeline

The MERS audio emotion recognition model is trained using a robust pipeline that aggregates multiple datasets, applies augmentation, and optimizes hyperparameters.

### Data Sources
We utilize four standard datasets located in `datasets/`:
- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song.
- **TESS**: Toronto emotional speech set.
- **SAVEE**: Surrey Audio-Visual Expressed Emotion.
- **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset.

### Preprocessing & Feature Extraction
1.  **Loading**: Audio files are loaded at 16kHz sample rate.
2.  **Augmentation**: To improve robustness, we apply:
    -   **Noise Injection**: Adding random Gaussian noise.
    -   *(Optional)* Pitch Shifting and Time Stretching (configurable in `train_full_model.py`).
3.  **Feature Extraction**:
    -   **Prosody**: RMS Energy (Mean/Std), Zero Crossing Rate, Spectral Centroid.
    -   **Spectral**: MFCCs (13 coefficients).
    -   **Temporal**: Approximate Speaking Rate.
    -   **Total Dimensions**: ~18 features per sample.
4.  **Scaling**: Features are standardized using `StandardScaler` (saved as `audio_scaler.pkl`).

### Model Architecture & Training
-   **Model**: Multi-Layer Perceptron (MLP) Classifier (Scikit-learn).
-   **Hyperparameter Tuning**: We use `RandomizedSearchCV` to optimize:
    -   `hidden_layer_sizes`: e.g., (128, 64), (256, 128).
    -   `activation`: ReLU, Tanh.
    -   `solver`: Adam, SGD.
    -   `alpha`: L2 regularization term.
    -   `learning_rate`: Constant vs Adaptive.
-   **Evaluation**: The model is evaluated on a stratified 20% test set using Accuracy and Classification Report (Precision, Recall, F1-score).

## 2. Model Performance Metrics

### Audio MLP Model (Scikit-learn)
- **Training Date**: 2026-02-07
- **Dataset**: Combined Crema, RAVDESS, SAVEE, TESS (Augmented)
- **Total Samples**: ~26,000
- **Test Accuracy**: **80.61%**

#### Hyperparameters (Best Found)
- **Architecture**: MLP (256, 128)
- **Activation**: ReLU
- **Solver**: Adam
- **Learning Rate**: Adaptive
- **Alpha**: ~0.00015

#### Classification Report
| Emotion | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Focused** | 0.91 | 0.91 | **0.91** | 314 |
| **Angry** | 0.83 | 0.85 | 0.84 | 822 |
| **Sad** | 0.84 | 0.82 | 0.83 | 822 |
| **Neutral** | 0.83 | 0.79 | 0.81 | 825 |
| **Happy** | 0.79 | 0.76 | 0.78 | 822 |
| **Fear** | 0.78 | 0.79 | 0.78 | 822 |
| **Stressed** | 0.73 | 0.79 | 0.76 | 822 |

*Note: "Focused" achieves the highest performance, likely due to distinct spectral features in "Surprise/Focused" samples.*

## 3. Interface Integration Architecture

The system follows a Local Client-Server Architecture.

### Components
1.  **Backend (Laptop/PC)**:
    -   **FastAPI Server**: Exposes REST endpoints.
    -   **Engines**:
        -   `VisualEngine`: Runs MediaPipe Face Mesh on local webcam feed (Threaded).
        -   `AudioEngine`: Processes incoming audio using the trained MLP model.
        -   `FusionEngine`: Combines predictions with "Audio Priority" logic.
        -   `XAIEngine`: Generates human-readable explanations.
2.  **Clients**:
    -   **Mobile App (Flutter)**: Captures audio, sends to server, displays results.
    -   **Desktop Dashboard (Python/Tkinter)**: Displays camera feed, graphs, and detailed analytics.

### Data Flow (Mobile Request)
1.  **Input**: User speaks into Mobile App. App records audio (WAV).
2.  **Request**: App sends POST `/analyze_multimodal` with audio file.
3.  **Processing**:
    -   **Audio**: Server processes uploaded file -> Audio Emotion & Confidence.
    -   **Visual**: Server retrieves *latest* frame analysis from the active Webcam thread -> Visual Emotion & Confidence.
    -   **Fusion**: `FusionEngine` combines them. If visual confidence is low (<0.5) or audio is stronger, Audio wins.
4.  **Response**: JSON payload returned to App.

### JSON Output Schema
The API is strictly typed to ensure UI consistency.

```json
{
  "emotion": "Focused",
  "confidence": 0.85,
  "color": "#1ABC9C",
  "modality_weights": {
    "audio": 0.7,
    "visual": 0.3
  },
  "explanation": {
    "visual": "Eyebrows lowered and minimal lip movement",
    "audio": "Steady speaking rate and stable pitch"
  },
  "timestamp": "2026-02-07T12:15:30",
  "debug": { ... }
}
```

## 4. Production Deployment

### Requirements
-   Python 3.9+
-   Dependencies: `requirements.txt`
-   Local Network: PC and Mobile must be on the same Wi-Fi/Hotspot.

### Running the Server
```bash
python mers/server.py
```
*Server runs on 0.0.0.0:8000*

### Running the Dashboard
```bash
python dashboard.py
```
*Launches GUI + Internal Server*

### Running the Mobile App
Build APK:
```bash
cd mers_app
flutter build apk --release
```
Install `build/app/outputs/flutter-apk/app-release.apk` on Android device.
```