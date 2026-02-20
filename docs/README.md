# MERS: Multimodal Emotion Recognition System

MERS is a local, privacy-preserving multimodal emotion recognition framework that analyzes facial expressions and vocal characteristics simultaneously. It is designed to run entirely offline with a client-server architecture.

## Key Features
- **Privacy-First**: No cloud services; all processing happens on your local machine.
- **Client-Server Architecture**:
  - **Backend**: Python (FastAPI + Scikit-learn) running on Laptop/PC.
  - **Client**: Mobile App (Flutter) or Desktop Script.
- **Lightweight AI**:
  - **Visual**: MediaPipe Face Mesh (Geometric Features).
  - **Audio**: Scikit-learn MLP (Prosody & Spectral Features).
- **Explainable AI (XAI)**: Provides human-readable reasons for detections (e.g., "High pitch and smile detected").

## System Architecture

1.  **Backend (PC)**: Extracts features from Audio/Video and runs ML inference.
2.  **Mobile Client (Android)**: Records audio/video and sends it to the Backend via Wi-Fi.

## Installation

1. **Clone/Download** the repository.
2. **Install Dependencies** (Python 3.9+):
   ```bash
   pip install -r mers/requirements.txt
   ```

## Usage

### 1. Start the Backend Server
Run this on your Laptop/PC:
```bash
python mers/server.py
```
*The server will listen on `0.0.0.0:8000` (All interfaces).*

### 2. Run the Client

**Option A: Mobile App (Flutter)**
See [mers_app/BUILD_APK.md](mers_app/BUILD_APK.md) for full instructions on building and installing the APK.
1.  Connect phone to the same Wi-Fi.
2.  Enter Laptop's IP (e.g., `192.168.1.5`) in the app.
3.  Press & Hold to record.

**Option B: Desktop Client (Testing)**
Run this in a separate terminal to test without a phone:
```bash
python client_pc.py
```

## Training (Optional)

To retrain the lightweight Scikit-learn models:
```bash
python mers/train/train_audio_sklearn.py
```
*(Requires datasets in `datasets/` folder)*

## Project Structure
```
mers/
├── mers/
│   ├── server.py           # Main FastAPI Backend
│   ├── src/                # Core Engines (Audio, Visual, Fusion)
│   ├── models/             # Saved Scikit-learn Models (.pkl)
│   └── train/              # Training Scripts
├── mers_app/               # Flutter Mobile App Source Code
├── client_pc.py            # Desktop Testing Client
└── datasets/               # Training Data
```

## License
MIT License.
