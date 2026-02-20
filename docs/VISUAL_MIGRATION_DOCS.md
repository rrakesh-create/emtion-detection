# MERS Visual Module Upgrade: Migration to Deep Learning

## 1. Problem Analysis
The initial MERS implementation used a rule-based heuristic approach (`VisualEngine`) based on Facial Action Coding System (FACS) logic.
- **Issue**: Users reported incorrect emoji mappings, specifically for "Stressed" (Disgust) and "Sad".
- **Root Cause**: Heuristic thresholds for Action Units (e.g., "brow squeeze" for Anger vs "nose scrunch" for Disgust) are brittle and vary significantly across individuals and lighting conditions.
- **Solution**: Replace heuristics with a Data-Driven approach using a Convolutional Neural Network (CNN).

## 2. Dataset Selection: FER2013
We selected the **FER2013** dataset (Facial Expression Recognition 2013) for the following reasons:
- **Size**: 35,887 grayscale images (48x48 pixel faces).
- **Classes**: 7 standard emotions that align well with MERS requirements.
- **Availability**: Publicly available via Kaggle.

### Emotion Mapping
We map the standard FER2013 classes to MERS output emotions as follows:

| FER2013 Class | MERS Emotion | MERS Emoji | Logic |
| :--- | :--- | :--- | :--- |
| `angry` | **Angry** | 😡 | Direct Match |
| `disgust` | **Stressed** | 😫 | Disgust is a strong proxy for high stress/aversion |
| `fear` | **Fear** | 😨 | Direct Match |
| `happy` | **Happy** | 😄 | Direct Match |
| `neutral` | **Neutral** | 😐 | Direct Match |
| `sad` | **Sad** | 😢 | Direct Match |
| `surprise` | **Focused** | 😮 | Surprise indicates high alertness/focus |

## 3. Model Architecture: EfficientNet-B0
We chose **EfficientNet-B0** over MobileNet or ResNet due to its superior accuracy-to-parameter ratio.
- **Input**: 224x224 RGB Images (Resized from 48x48 FER images).
- **Pre-training**: Initialized with ImageNet weights to leverage feature extraction.
- **Classifier**: Custom top layer outputting 7 classes.

## 4. Implementation Details
- **Training Script**: `train_visual_cnn.py`
  - Optimizes for GPU (CUDA).
  - Uses Data Augmentation (Rotation, Flip) to prevent overfitting.
  - Saves best model to `mers/models/visual_efficientnet.pth`.
- **Inference Engine**: `VisualEngineCNN` (`mers/src/visual_engine_cnn.py`)
  - Inherits from `VisualEngine` to keep MediaPipe Face Detection.
  - Crops the detected face, resizes, and feeds it to the CNN.
  - Returns probabilities mapped to the MERS schema.

## 5. Validation Protocol
After training, run `evaluate_model.py` to generate:
1.  **Confusion Matrix**: Visualizes misclassifications (saved to `mers/reports/confusion_matrix.png`).
2.  **Classification Report**: Precision, Recall, and F1-Score for each emotion.

**Success Criteria**:
- >80% Accuracy on "Happy", "Neutral", "Sad".
- >70% Accuracy on "Stressed" (Disgust) and "Focused" (Surprise).
