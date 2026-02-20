# Visual Dataset Analysis & Migration Plan

## 1. Executive Summary
**Issue**: Users report incorrect emojis displaying for specific emotional states.
**Root Cause**: The current visual implementation does **not** use a trained dataset/model. It relies on **heuristic rules** (expert system) based on MediaPipe Face Mesh landmarks. These rules are brittle, sensitive to calibration, and lack the generalization of deep learning models.
**Recommendation**: Migrate from rule-based heuristics to a Deep Learning model (CNN) trained on a robust dataset like **AffectNet** or **RAF-DB**.

---

## 2. Current Implementation Analysis
### Architecture
- **Engine**: `VisualEngine` (MediaPipe FaceLandmarker)
- **Logic**: Geometric feature extraction (e.g., `brow_raise`, `mouth_open`) + If/Else Thresholds.
- **Dataset**: **None**. No training data was used.
- **Performance**:
  - *Pros*: Fast, CPU-efficient, explainable (easy to map to XAI).
  - *Cons*: High false positive rate for "Fear" vs "Surprise"; "Neutral" often misclassified if face is not perfectly still.

### Why Emojis are Incorrect
The "Emojis" are simple UI mappings of the predicted text label. If the rule-based engine predicts "Happy" because the user's mouth is naturally wide (even if not smiling), the UI displays "😊". The error is in the **prediction logic**, not the UI mapping.

---

## 3. Alternative Datasets Evaluation

| Dataset | Size | Resolution | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FER2013** | ~35,887 | 48x48 (Gray) | Standard benchmark, small size. | Low res, noisy labels, unrealistic poses. | **Low** |
| **CK+** | ~593 | High | Clean lab data. | Too small, posed expressions (not natural). | **No** |
| **RAF-DB** | ~30,000 | High (Color) | Real-world, high accuracy. | Smaller than AffectNet. | **High** |
| **AffectNet** | ~450,000 | High (Color) | Largest "in-the-wild" dataset. | Large size (requires more compute to train). | **Highest** |

**Selected Target**: **AffectNet** (or RAF-DB as a lighter alternative).
*Rationale*: AffectNet covers the widest range of head poses, lighting, and ethnicities, crucial for a robust local desktop application.

---

## 4. Migration Plan

### Phase 1: Data Acquisition & Preparation
1.  **Download**: Acquire AffectNet (requires request) or RAF-DB.
2.  **Preprocessing**:
    - Detect faces using MediaPipe (to match production pipeline).
    - Crop and align faces to 224x224.
    - Normalization: Mean/Std (ImageNet standards).
3.  **Label Mapping**:
    - Map Dataset Labels -> MERS Output Emotions (`Focused`, `Stressed` will need careful mapping, e.g., Surprise->Focused, Disgust->Stressed).

### Phase 2: Model Development
1.  **Architecture**: **EfficientNet-B0** or **MobileNetV3-Large** (balance of speed/accuracy).
2.  **Training Strategy**:
    - Transfer Learning (ImageNet weights).
    - Augmentation: RandomRotate, ColorJitter, GaussianNoise (simulate webcam quality).
    - Loss Function: CrossEntropy (potentially weighted for class imbalance).
3.  **Output**: Export to **ONNX** for fast CPU inference in Python.

### Phase 3: Integration
1.  **Replace Logic**:
    - Modify `VisualEngine` to load the ONNX model.
    - Remove `_extract_features` and `_detect_emotion_rules`.
    - Keep `FaceLandmarker` only for face detection/cropping (or switch to lighter detector).
2.  **Emoji Validation**:
    - Update `config.py` mappings if model classes change.

### Phase 4: Verification
1.  **Run `validate_visual.py`**:
    - Benchmark against a held-out test set.
    - **Success Metric**: >85% Accuracy on Test Set.
2.  **User Acceptance**:
    - Verify Emojis match user expressions in real-time Dashboard.

---

## 5. Validation Framework
Use the provided `validate_visual.py` script.

**Usage**:
```bash
python validate_visual.py --dataset C:/Datasets/RAF-DB/test
```

**Metrics Tracked**:
- Confusion Matrix (Heatmap)
- Precision/Recall per Emotion
- Emoji Display Accuracy (Direct hit rate)

## 6. Conclusion
The current "Emoji Error" is a symptom of the underlying rule-based architecture. Replacing it with a CNN trained on AffectNet/RAF-DB is the only viable long-term solution to achieve professional-grade accuracy (>85%).
