# MERS Visual Module Migration Plan

## 1. Executive Summary
The current MERS visual module uses a rule-based heuristic approach (Action Units via MediaPipe) which lacks robustness and fails to align with standard emotion definitions (e.g., Happy, Sad, Angry). To resolve "incorrect emoji" issues and improve accuracy, we are migrating to a Deep Learning approach using a CNN (EfficientNet-B0) trained on the FER2013 dataset.

## 2. Current State Assessment
- **Method**: Heuristic Action Unit scoring (no learned weights).
- **Dataset**: None (Logic-based).
- **Performance**: Low accuracy on standard benchmarks (see `mers/evaluation/classification_report.txt`).
- **Issues**: Misalignment between detected landmarks and user-expected emotions (e.g., Surprise mapped to Focused causing confusion).

## 3. Migration Strategy

### Phase 1: Infrastructure (In Progress)
- [x] Enable GPU support (Installing PyTorch CUDA 12.6).
- [x] Download FER2013 dataset.
- [x] Establish baseline performance (`benchmark_current_logic.py`).

### Phase 2: Model Training
- [ ] Train EfficientNet-B0 on FER2013.
- **Script**: `train_visual_cnn.py`
- **Config**: 
  - Batch Size: 32
  - Epochs: 5 (Initial), 20 (Production)
  - Optimizer: Adam (lr=0.001)

### Phase 3: Integration
- [ ] Create `VisualEngineCNN` class to replace/augment `VisualEngine`.
- [ ] Load `mers/models/visual_efficientnet.pth`.
- [ ] Update `dashboard.py` to use the new engine.

### Phase 4: Validation
- [ ] Run validation script on test set.
- [ ] Target Accuracy: >65% (State of the art on FER2013 is ~70-73%).
- [ ] Verify Emoji mappings align with model predictions.

## 4. Emoji Mapping Update
Ensuring the trained model classes map correctly to user feedback:
- 0: Angry -> 😠
- 1: Disgust -> 😫 (Stressed)
- 2: Fear -> 😨
- 3: Happy -> 😊
- 4: Neutral -> 😐
- 5: Sad -> 😢
- 6: Surprise -> 🧐 (Focused)

## 5. Rollback Plan
Keep `_detect_emotion_rules` as a fallback method in `VisualEngine` if the model fails to load or inference is too slow on CPU-only edge cases.
