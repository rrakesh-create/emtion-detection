# MERS: Multimodal Emotion Recognition System

## 1. Project Overview
This project presents an adaptive and explainable multimodal emotion recognition system (MERS) that integrates facial analysis, speech-based emotion detection, and user interaction to accurately identify human emotional states in real-time.

The core innovation lies in its privacy-preserving architecture and explainability layer. Unlike traditional systems that process raw video feeds, MERS analyzes geometric relationships between facial landmarks and vocal prosody features to infer emotions.

## 2. Key Features & Implementation Details

### A. adaptive Visual Analysis (Landmark-Based)
- **Face Mesh Technology**: Utilizes Google MediaPipe to extract 468+ facial landmarks in real-time.
- **Geometric Feature Extraction**: Instead of using Convolutional Neural Networks (CNNs) on raw pixels, the system computes relative distances and angles between key facial regions (eyes, eyebrows, lips, nose, jawline).
- **Privacy-First**: No facial images are stored or transmitted. The system processes only coordinate data, ensuring user privacy.
- **Visualization**: Real-time overlay of facial landmarks (lines and points) demonstrating the system's focus areas.

### B. Audio-First Emotion Recognition
- **Vocal Prosody Analysis**: Extracts acoustic features such as pitch, energy, speaking rate, pause duration, and voice tremor.
- **Dynamic Prioritization**: The system dynamically assigns higher importance to audio cues when facial expressions are neutral, ambiguous, or when lighting conditions are poor.

### C. Adaptive Multimodal Fusion
- **Confidence-Weighted Integration**: Outputs from the Visual and Audio modules are combined using an adaptive fusion strategy.
- **Reliability Scoring**: Weights are adjusted in real-time based on the confidence level of each modality (e.g., if the face is partially occluded, audio weight increases).

### D. Explainable AI (XAI) Layer
- **Transparent Decision Making**: The system provides clear, human-readable explanations for its predictions.
- **Example Explanations**:
  - *"Detected 'Happy' because: Lip corners are raised (Visual) and Pitch is elevated (Audio)."*
  - *"Detected 'Focus' because: Eyebrows are lowered (Visual) and Speaking rate is steady."*

## 3. System Architecture: Local Client-Server

The system adopts a **Local Client-Server Architecture** designed for robust, offline-capable deployment without cloud dependencies.

### Backend (The "Brain")
- **Platform**: High-performance Laptop/PC.
- **Technology**: Python, FastAPI.
- **Role**:
  - Runs the heavy Machine Learning models (Landmark MLP, Audio Model).
  - Handles the Fusion Logic and Explainability generation.
  - Exposes REST API endpoints (e.g., `/analyze_multimodal`) for the client.

### Client (The "Interface")
- **Platform**: Companion Mobile Application (Flutter/React Native).
- **Role**:
  - Captures Audio and transmits it to the backend.
  - Displays the real-time Emotion Prediction and Explanation.
  - Acts as a lightweight UI, minimizing battery and processor usage on the mobile device.

### Communication
- **Network**: Local Wi-Fi / Hotspot.
- **Protocol**: HTTP/REST over local IP (e.g., `http://192.168.1.5:8000`).

## 4. Workflow
1.  **User Interaction**: User speaks or interacts with the camera.
2.  **Data Capture**:
    *   **Visual**: Laptop webcam captures video -> Extracts Landmarks.
    *   **Audio**: Mobile App (or Laptop mic) captures audio -> Sends to Backend.
3.  **Processing (Backend)**:
    *   **Visual Engine**: Computes geometry from landmarks -> Predicts visual emotion.
    *   **Audio Engine**: Extracts prosody -> Predicts audio emotion.
4.  **Fusion & Explanation**: The Fusion Engine combines predictions and generates a textual explanation.
5.  **Output**: Results are displayed on the Dashboard (Laptop) and Mobile App (Client).
