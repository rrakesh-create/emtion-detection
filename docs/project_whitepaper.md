# Project: Multimodal Emotion Recognition System (MERS)

## 1. Abstract
This project presents MERS, a robust Multimodal Emotion Recognition System designed to detect human emotions from both Audio (Speech) and Visual (Facial Expressions) inputs. The system specifically addresses low-resource languages by implementing a synthetic data generation pipeline for Telugu speech. It employs a **ResNet-18** architecture for audio emotion recognition (achieving **93.66% accuracy**) and an **EfficientNet-B0** CNN for facial expression analysis. A late-fusion strategy combines these modalities with adaptive weighting to provide a final, confident emotion prediction, enhanced by an Explainable AI (XAI) module that interprets facial landmarks.

## 2. Introduction
Emotion recognition is a cornerstone of advanced Human-Computer Interaction (HCI). While visual emotion recognition is well-studied, audio emotion recognition for Indian languages like Telugu remains under-resourced. MERS bridges this gap by:
1.  **Audio:** Creating a large synthetic Telugu dataset and training a high-performance Deep Learning model.
2.  **Visual:** Utilizing state-of-the-art CNNs and Face Mesh technology for real-time facial analysis.
3.  **Fusion:** Integrating both streams to improve reliability, robustness, and explainability.

## 3. Audio Modality (Telugu Speech)

### 3.1. Dataset Construction
To overcome data scarcity, we developed a comprehensive dataset:
*   **Original Source:** Kaggle Telugu Emotion Speech dataset.
*   **Synthetic Expansion (TTS):** Used Google Text-to-Speech (gTTS) to generate 100+ new samples across 5 emotions, ensuring linguistic diversity.
*   **Data Augmentation:** Applied `audiomentations` (Gaussian Noise, Time Stretch, Pitch Shift) with a **5x augmentation factor**, growing the dataset to ~3,000 samples.

### 3.2. Audio Model Architecture
*   **Model:** **ResNet-18** (Residual Neural Network).
*   **Input:** Log-Mel Spectrograms (Audio converted to visual representations).
    *   *Parameters:* Sample Rate: 16kHz, Duration: 3s, n_mels: 128.
*   **Adaptation:** The first convolutional layer was modified to accept 1-channel input (spectrograms), and the fully connected layer was adapted for 5 emotion classes (`Angry`, `Happy`, `Neutral`, `Sad`, `Surprised`).
*   **Training:** Optimized using AdamW with a Learning Rate Scheduler to achieve optimal convergence.

## 4. Visual Modality (Facial Expressions)

### 4.1. Facial Recognition & Detection
The system processes video input in real-time to detect and analyze faces:
*   **Face Detection:** Uses **MediaPipe Face Detection** (BlazeFace), an ultra-lightweight detector optimized for mobile and edge devices. It locates the face bounding box in milliseconds.
*   **Feature Extraction:** Uses **MediaPipe Face Mesh** to map **468 3D facial landmarks**. These landmarks provide a dense geometric representation of the face, tracking subtle movements of the eyes, eyebrows, lips, and jaw.

### 4.2. Visual Model Architecture (CNN)
The core classification is performed by a Deep Convolutional Neural Network:
*   **Model:** **EfficientNet-B0**.
*   **Architecture:** EfficientNet uses a compound scaling method that balances depth, width, and resolution for maximum efficiency and accuracy.
*   **Input:** Cropped face images resized to 224x224 pixels.
*   **Output:** Probabilities for 7 emotions: `Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`.
*   **Training:** Transfer learning from ImageNet, fine-tuned on large-scale facial expression datasets (e.g., FER2013).

### 4.3. Criteria for Classification (Geometric Feature Analysis)
To ensure the system is interpretable, we use a rule-based layer alongside the CNN. This layer analyzes the **468 facial landmarks** extracted by MediaPipe to compute specific **Action Units (AUs)**.

**A. Core Normalization**
All distances are normalized by the **Face Width** to ensure invariance to camera distance (scale invariance).
*   `Scale = Distance(Left Cheek, Right Cheek)`
*   `Normalized Distance = Raw Distance / Scale`

**B. Key Landmarks & Formulas**
The system tracks specific indices on the 468-point mesh:

| Feature | Landmarks Used (Index) | Formula Description | associated AU |
| :--- | :--- | :--- | :--- |
| **Brow Raise** | 65 (L.Brow), 159 (L.Eye) | `Dist(Brow, Eye) / Scale` | AU1 + AU2 |
| **Brow Squeeze** | 55 (L.Inner), 285 (R.Inner) | `Dist(L.Inner, R.Inner) / Scale` | AU4 |
| **Eye Openness** | 159 (Top), 145 (Bottom) | `Dist(Top, Bottom) / Scale` | AU5 (Wide) / AU6 (Squint) |
| **Mouth Width** | 61 (L.Corner), 291 (R.Corner) | `Dist(L.Corner, R.Corner) / Scale` | AU20 (Stretch) |
| **Smile Curve** | 13 (Upper Lip), Corners | `(LipY - Avg(CornerY)) / Scale` | AU12 (Smile) |
| **Mouth Open** | 13 (Upper Lip), 14 (Lower Lip) | `Dist(Upper, Lower) / Scale` | AU25/26 |

**C. Detailed Emotion Breakdown (What is What)**

Here is a detailed explanation of the visual criteria for each emotion:

1.  **Happy (Joy)**
    *   **Visual Cue:** Smiling and "smiling with eyes".
    *   **What Happens:** The **Zygomatic Major** muscle pulls the corners of the mouth up and back (Smile Curve > 0.01). Simultaneously, the **Orbicularis Oculi** muscle contracts around the eyes, raising the cheeks and narrowing the eye opening (Eye Openness < 0.055), creating "crow's feet".
    *   **Technical Term:** Action Unit 12 (Lip Corner Puller) + Action Unit 6 (Cheek Raiser).

2.  **Sad (Sadness)**
    *   **Visual Cue:** Frowning and drooping features.
    *   **What Happens:** The **Depressor Anguli Oris** muscle pulls the corners of the mouth downwards (Smile Curve < -0.01). The inner eyebrows are raised and pulled together (Brow Height < 0.07 but Inner Brow raised), creating a characteristic "inverted V" shape.
    *   **Technical Term:** Action Unit 15 (Lip Corner Depressor) + Action Unit 1 (Inner Brow Raiser).

3.  **Angry (Anger)**
    *   **Visual Cue:** Glaring and tightening.
    *   **What Happens:** The **Corrugator Supercilii** muscle pulls the eyebrows down and together (Brow Squeeze Distance < 0.15). The eyelids tighten (Eye Squint), and the lips may be pressed together or bared in a snarl.
    *   **Technical Term:** Action Unit 4 (Brow Lowerer) + Action Unit 5 (Upper Lid Raiser) + Action Unit 23 (Lip Tightener).

4.  **Surprise (Surprise)**
    *   **Visual Cue:** Wide eyes and dropped jaw.
    *   **What Happens:** The **Frontalis** muscle raises the entire eyebrow arch high up (Brow Height > 0.10). The upper eyelids are raised to expose more of the eye white (Eye Openness > 0.075). The jaw drops open vertically (Mouth Open > 0.03) in a relaxed manner.
    *   **Technical Term:** Action Unit 1+2 (Inner+Outer Brow Raiser) + Action Unit 5 (Upper Lid Raiser) + Action Unit 26 (Jaw Drop).

5.  **Fear (Fear)**
    *   **Visual Cue:** Wide eyes and stretched mouth.
    *   **What Happens:** Similar to Surprise, the eyebrows are raised (Brow Height > 0.09) but are flatter. The key difference is the mouth: the **Risorius** muscle stretches the lips horizontally (Mouth Width > 0.45) in a grimace, rather than hanging open.
    *   **Technical Term:** Action Unit 1+2 (Brow Raiser) + Action Unit 20 (Lip Stretcher).

This geometric layer acts as a validation mechanism for the primary CNN predictions, providing the "Why" behind the classification.

**D. Geometric Thresholds (How Much Distance?)**

The system uses specific **Normalized Values** (where 1.0 = Face Width) to determine if a feature is active. Here are the exact thresholds used in the code:

| Feature | Condition | Threshold Value | Meaning |
| :--- | :--- | :--- | :--- |
| **Smile Curve** | **> 0.01** | Positive | Corners are lifted (Happy) |
| **Smile Curve** | **< -0.01** | Negative | Corners are drooped (Sad) |
| **Brow Squeeze** | **< 0.15** | Very Small Distance | Eyebrows are pulled together (Angry) |
| **Brow Height** | **> 0.10** | High Distance | Eyebrows are raised high (Surprise) |
| **Brow Height** | **< 0.07** | Low Distance | Eyebrows are lowered/relaxed (Angry/Sad) |
| **Eye Openness** | **> 0.075** | Large Opening | Eyes are wide open (Surprise/Fear) |
| **Eye Openness** | **< 0.055** | Small Opening | Eyes are squinting (Happy) |
| **Mouth Width** | **> 0.45** | Wide Distance | Lips are stretched horizontally (Fear) |
| **Mouth Open** | **> 0.03** | Vertical Opening | Jaw is dropped (Surprise) |

*Example:* If the distance between your inner eyebrows is **0.12** (relative to face width), it is less than the **0.15** threshold, so the system classifies it as "Brow Squeeze" (Angry).

**E. Strategies for Improvement (How to Make it "The Best")**

To push the accuracy from standard (FER2013 level ~65-70%) to state-of-the-art (>85%), the following advanced techniques can be implemented:

1.  **Dataset Upgrade (The Biggest Impact)**
    *   **Current:** Most models train on FER2013 (messy, low-res, grayscale).
    *   **Improvement:** Fine-tune on **AffectNet** (400k+ images) or **RAF-DB** (Real-world Affective Faces). These datasets are cleaner, higher resolution, and more diverse.
    *   *Result:* Immediate 10-15% jump in real-world accuracy.

2.  **Model Architecture (Deeper & Smarter)**
    *   **Current:** EfficientNet-B0 (Lightweight, ~5.3M params).
    *   **Improvement:** Upgrade to **EfficientNet-B4** (Heavy, 19M params) or **Vision Transformers (ViT)**. ViTs are excellent at capturing global facial context rather than just local features.
    *   *Trade-off:* Slower inference speed (requires GPU).

3.  **Temporal Smoothing (Video vs Image)**
    *   **Current:** Frame-by-frame prediction (jittery).
    *   **Improvement:** Implement a **Rolling Average Window** (e.g., average predictions over the last 10 frames). Or use **LSTM/GRU** layers on top of the CNN to learn temporal dynamics (a smile forms over time, it doesn't just appear).
    *   *Result:* Eliminates "flickering" between emotions.

4.  **Personalized Calibration (The "Pro" Move)**
    *   **Concept:** Everyone's neutral face is different (some look naturally angry or sad).
    *   **Improvement:** Add a "Calibration Phase" where the user sits still for 5 seconds. The system records *their* specific baseline distances (e.g., their normal brow height).
    *   **Logic:** `Feature = (Current_Value - User_Baseline)`. This accounts for individual facial structure.

## 5. Multimodal Fusion
The system employs a **Decision-Level (Late) Fusion** strategy using an Adaptive Weighting mechanism (`FusionEngine`):

1.  **Confidence Extraction:** Extracts probability scores from both Audio (ResNet) and Visual (EfficientNet) models.
2.  **Adaptive Weighting:**
    *   Weights are dynamically adjusted based on confidence levels.
    *   *Heuristic 1 (Audio Reliability):* If Audio predicts "Angry" but energy (RMS) is low (background noise), its weight is reduced.
    *   *Heuristic 2 (Visual Reliability):* If the Visual model is unsure (<30% confidence), its prediction is weighted less.
3.  **Combination:** `Final_Score = (w_v * Visual_Score) + (w_w * Audio_Score)`
4.  **Result:** The emotion with the highest combined score is selected as the final prediction.

## 6. Results
*   **Audio (Speech):** The ResNet-18 model achieved a test accuracy of **93.66%**, proving the efficacy of the synthetic data generation pipeline.
*   **Visual (Face):** The EfficientNet-B0 model provides robust real-time performance (>30 FPS) on standard hardware.
*   **System:** The multimodal fusion creates a resilient system that can maintain high accuracy even when one modality fails (e.g., silent face or audio without video).

## 7. Implementation
The project is implemented in Python using:
*   **Deep Learning:** PyTorch (ResNet/EfficientNet training & inference).
*   **Computer Vision:** OpenCV, MediaPipe (Face Mesh).
*   **Audio Processing:** Librosa, Audiomentations.
*   **Interface:** Tkinter (Dashboard) and FastAPI (Mobile Server).

## 8. Future Roadmap (The "Wow" Factor)

To transform MERS from a recognition system into a complete **Empathetic AI Assistant**, the following features are planned:

### 8.1. Interactive Empathetic Response
Instead of just *detecting* emotions, the system will *respond* to them:
*   **Sadness Detected:** Play soothing music or offer a comforting quote.
*   **Anger Detected:** Suggest a breathing exercise or a calm-down technique.
*   **Happy Detected:** "Celebration Mode" with confetti animation on the dashboard.

### 8.2. Long-term Emotion Tracking
*   **Database Integration:** Store session data in a local SQLite database.
*   **Mood Dashboard:** Show a weekly "Mood Graph" (e.g., "You were 70% Happy this week").
*   **Utility:** Useful for therapists or self-reflection.

### 8.3. Cross-Platform Mobile App
*   **Tech Stack:** Flutter or React Native.
*   **Deployment:** Package the Python backend as an API and deploy to the cloud (e.g., Hugging Face Spaces).
*   **Benefit:** Allows users to use MERS anywhere, not just on a PC.

### 8.4. Visual & Audio Explanations (Advanced XAI)
*   **Visual:** Show a heat map on the face indicating *exactly* which muscles triggered the emotion.
*   **Audio:** Highlight the specific frequency bands in the spectrogram that match the detected tone.
