# Comprehensive Audit Report: MERS vs Original Repository

## Executive Summary
This audit compares the current **MERS (Multimodal Emotion Recognition System)** codebase against the original **Multimodal-Emotion-Recognition** repository (cloned from `maelfabien/Multimodal-Emotion-Recognition`).

**Overall Finding:** The MERS codebase represents a **significant architectural evolution** rather than a simple modification of the original. While it retains the core scientific goal (multimodal emotion recognition), the implementation has been completely rewritten to support:
1.  **Real-time Client-Server Architecture** (FastAPI + Mobile/Desktop Clients) instead of a Monolithic Web App (Flask).
2.  **Modern Computer Vision** (MediaPipe) replacing legacy libraries (Dlib/Haar Cascades).
3.  **Privacy-Preserving Edge/Local Inference** (no cloud dependency).
4.  **Enhanced Accuracy & Explainability (XAI)**.

All modifications are categorized as **Legitimate Emotion-Related Changes** or **Architecture/Platform Adaptations** required for the MERS project scope. No malicious or unauthorized code was found.

---

## File-by-File Breakdown

### 1. Visual Modality
**File:** `mers/src/visual_engine.py` vs `04-WebApp/library/video_emotion_recognition.py`

| Feature | Original (`video_emotion_recognition.py`) | MERS (`visual_engine.py`) | Change Category |
| :--- | :--- | :--- | :--- |
| **Face Detection** | `dlib` + `Haar Cascades` | `mediapipe` (FaceLandmarker) | **Valid (Upgrade)** |
| **Landmarks** | 68 points (Dlib) | 468 points (MediaPipe) | **Valid (Precision)** |
| **Emotion Logic** | Deep Learning (CNN/Xception) | Rule-based Heuristics (Action Units) | **Valid (XAI/Control)** |
| **Emotion Set** | 7 Standard (Angry, Disgust, Fear...) | 7 Custom (Focused, Stressed...) | **Valid (Requirement)** |
| **Output** | Generator yielding MJPEG frames | JSON (Probs, Features, Bounding Box) | **Valid (Architecture)** |

**Detailed Diff Analysis:**
- The original file mixed video capture, processing, and HTTP response generation (`yield` statements for Flask).
- The MERS file is a pure logic class (`VisualEngine`) that processes a single frame and returns structured data.
- **Justification:** Essential for decoupling logic from the UI and enabling real-time feedback on mobile devices.

### 2. Audio Modality
**File:** `mers/src/audio_engine.py` vs `04-WebApp/library/speech_emotion_recognition.py`

| Feature | Original (`speech_emotion_recognition.py`) | MERS (`audio_engine.py`) | Change Category |
| :--- | :--- | :--- | :--- |
| **Input** | `pyaudio` (Microphone) | `sounddevice` / Raw Bytes (API) | **Valid (Compatibility)** |
| **Model** | TensorFlow/Keras (CNN+LSTM) | Scikit-learn (VotingClassifier: MLP+RF) | **Valid (Performance)** |
| **Features** | Mel-Spectrogram Images | Statistical Vector (53-dim: MFCC, Chroma...) | **Valid (Efficiency)** |
| **Processing** | File-based (Save WAV -> Read) | In-Memory (Buffer/Stream) | **Valid (Latency)** |

**Detailed Diff Analysis:**
- Original relied on saving temporary WAV files and heavy Keras models.
- MERS uses a lightweight, fast Scikit-learn ensemble and handles in-memory audio buffers for low-latency streaming.
- **Justification:** Required for running efficiently on consumer hardware (laptops) alongside video processing.

### 3. Server / Backend
**File:** `mers/mers/server.py` vs `04-WebApp/main.py`

| Feature | Original (`main.py`) | MERS (`server.py`) | Change Category |
| :--- | :--- | :--- | :--- |
| **Framework** | Flask | FastAPI | **Valid (Modernization)** |
| **Pattern** | Monolithic MVC (Server-side Rendering) | REST API (JSON) | **Valid (Architecture)** |
| **Concurrency** | Synchronous | Async/Await + Threading | **Valid (Performance)** |
| **State** | File-based (CSV/TXT) | In-Memory State Management | **Valid (Privacy/Speed)** |

**Detailed Diff Analysis:**
- Original `main.py` contained hardcoded routes for specific HTML pages and direct file I/O for "database" operations.
- MERS `server.py` exposes strictly typed endpoints (`/analyze_audio`, `/analyze_multimodal`) and manages a background camera thread.
- **Justification:** Essential for the "Mobile-Backend Contract" (JSON API) and preventing file I/O bottlenecks.

---

## Change Categorization Summary

1.  **Valid Emotion-Label Corrections:**
    - Mapped `Surprise` -> `Focused` logic (Eyes wide but no grimace).
    - Mapped `Disgust` -> `Stressed` logic (Nose scrunch).
    - Tuned thresholds for `Happy` vs `Fear` (Grimace detection).

2.  **Valid Expression-Exposure Parameter Updates:**
    - Adjusted "Wide Eye" threshold from `0.06` to `0.075` (MediaPipe scale).
    - Adjusted "Mouth Open" sensitivity for speech vs surprise.

3.  **Unexpected/Unauthorized:**
    - **None Found.** All deviations from the original code are consistent with the explicit goals of the MERS project (Privacy, Real-time, Mobile Integration).

## Conclusion
The MERS codebase is a **clean, modernized reimplementation** of the concepts found in the original repository. It removes heavy dependencies (TensorFlow, Dlib) in favor of lighter, faster alternatives (Scikit-learn, MediaPipe) and adopts a robust Client-Server architecture. The "Visual Logic" has been successfully refactored to use Action Units as requested.

