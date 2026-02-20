"""
Visual Engine Module
Handles face detection and feature extraction using MediaPipe.
"""

import os
from typing import Dict, Tuple, Any, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config.settings import EMOTIONS, MODELS_DIR

class VisualEngine:
    """
    Visual Engine using MediaPipe Tasks API (FaceLandmarker).
    Extracts 468 landmarks and computes geometric features.
    """
    def __init__(self):
        print("[VisualEngine] Initializing MediaPipe FaceLandmarker...")
        
        # Resolve model path
        model_path = os.path.join(MODELS_DIR, "face_landmarker.task")
        
        if not os.path.exists(model_path):
            print(f"[VisualEngine] Error: Model not found at {model_path}")
            raise FileNotFoundError("face_landmarker.task not found. Please download it.")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE)
            
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # 1. Temporal Smoothing History
        from collections import deque
        self.history_len = 5 # Smooth over 5 frames (~0.15s at 30fps)
        self.history = deque(maxlen=self.history_len)
        
        print("[VisualEngine] Initialized with Smoothing (Window=5).")

    def process_frame(self, frame: np.ndarray) -> Tuple[
            Optional[np.ndarray], 
            Optional[Tuple[int, int, int, int]], 
            Dict[str, Any], 
            Any
        ]:
        """
        Process a single frame for emotion detection.

        Args:
            frame: Input image (BGR).

        Returns:
            - emotion_probs: np.array of probabilities matching EMOTIONS order
            - face_box: (x1, y1, x2, y2) bounding box
            - features: Dict of geometric features for XAI
            - landmarks: MediaPipe landmarks object
        """
        # Create MP Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect
        try:
            detection_result = self.detector.detect(mp_image)
        except Exception as e:
            print(f"Detection Error: {e}")
            return None, None, {}, None

        if not detection_result.face_landmarks:
            return None, None, {}, None

        # Get first face
        landmarks = detection_result.face_landmarks[0] # List of NormalizedLandmark

        # 1. Bounding Box (for UI)
        h, w, _ = frame.shape
        face_box = self._get_face_box(landmarks, w, h)

        # 2. Extract Geometric Features
        features = self._extract_features(landmarks, w, h)

        # 3. Rule-Based Emotion Detection
        probs = self._detect_emotion_rules(features)

        # Convert dict to array matching EMOTIONS list order
        prob_array = np.zeros(len(EMOTIONS))
        for i, emo in enumerate(EMOTIONS):
            prob_array[i] = probs.get(emo, 0.0)

        # 4. Temporal Smoothing
        smooth_probs = self._smooth_predictions(prob_array)

        return smooth_probs, face_box, features, landmarks

    def draw_overlays(self, frame, face_box, landmarks, emotion_label=None, confidence=None):
        """
        Draw bounding box, landmarks, and emotion label on the frame.
        """
        if frame is None:
            return None
            
        annotated_frame = frame.copy()
        h, w, _ = annotated_frame.shape

        # Draw Face Mesh
        if landmarks:
            # We need to reconstruct NormalizedLandmarkList for drawing_utils if we want full mesh
            # But we can just draw points for simplicity or use solutions.drawing_utils
            # Since landmarks is a list of NormalizedLandmark, we can use it directly if we had the wrapper
            # For now, let's draw the Bounding Box and key points
            pass
            
            # Draw Mesh Points (simplified)
            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_frame, (x, y), 1, (0, 255, 255), -1)

        # Draw Bounding Box
        if face_box:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Label
            if emotion_label:
                text = f"{emotion_label}"
                if confidence is not None:
                    text += f" ({confidence:.2f})"
                
                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 30), (x1 + text_w, y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        return annotated_frame

    def _get_face_box(self, landmarks, w, h):
        """Calculate bounding box from landmarks."""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x_min, x_max = min(xs) * w, max(xs) * w
        y_min, y_max = min(ys) * h, max(ys) * h
        return (int(x_min), int(y_min), int(x_max), int(y_max))

    def _extract_features(self, landmarks, w, h):
        """
        Compute Action Units (AUs) and geometric features.
        landmarks: List of NormalizedLandmark objects (x, y, z)
        """
        def get_pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        # Normalization Scale (Face Width)
        xs = [l.x for l in landmarks]
        face_width = (max(xs) - min(xs)) * w
        scale = face_width if face_width > 0 else 1.0

        # --- Geometric Measurements (Normalized) ---
        
        # 1. Brows
        # Inner Brow Height (relative to eye corners)
        left_brow_h = np.linalg.norm(get_pt(65) - get_pt(159)) / scale
        right_brow_h = np.linalg.norm(get_pt(295) - get_pt(386)) / scale
        avg_brow_h = (left_brow_h + right_brow_h) / 2.0
        
        # Brow Squeeze (Distance between inner brows)
        brow_dist = np.linalg.norm(get_pt(55) - get_pt(285)) / scale

        # 2. Eyes
        # Eye Openness (Upper lid to lower lid)
        left_eye_open = np.linalg.norm(get_pt(159) - get_pt(145)) / scale
        right_eye_open = np.linalg.norm(get_pt(386) - get_pt(374)) / scale
        avg_eye_open = (left_eye_open + right_eye_open) / 2.0

        # 3. Mouth
        # Mouth Width (Corner to corner)
        mouth_width = np.linalg.norm(get_pt(61) - get_pt(291)) / scale
        # Mouth Open (Upper lip to lower lip)
        mouth_open = np.linalg.norm(get_pt(13) - get_pt(14)) / scale
        
        # Smile/Frown (Corner Y vs Center Y)
        # Y increases downwards. 
        # Smile: Corners (small y) < Center (large y) -> Positive value
        # Frown: Corners (large y) > Center (small y) -> Negative value
        mouth_center_y = get_pt(13)[1]
        corners_y = (get_pt(61)[1] + get_pt(291)[1]) / 2.0
        smile_curve = (mouth_center_y - corners_y) / scale

        # 4. Nose
        # Nose Scrunch (Tip to Upper Lip)
        nose_lip_dist = np.linalg.norm(get_pt(1) - get_pt(13)) / scale

        # --- Action Unit (AU) Estimation (0.0 to 1.0 roughly) ---
        # These constants may need tuning, but this structure allows cleaner logic.
        
        features = {}
        
        # AU1/2: Brow Raise (Surprise/Fear)
        # Normal ~ 0.08. Raise > 0.10.
        features["au_brow_raise"] = max(0.0, (avg_brow_h - 0.08) * 10) 
        
        # AU4: Brow Lowerer (Anger)
        # Normal ~ 0.20. Squeeze < 0.15.
        features["au_brow_squeeze"] = max(0.0, (0.15 - brow_dist) * 10)
        features["au_brow_low"] = max(0.0, (0.07 - avg_brow_h) * 10)

        # AU5: Upper Lid Raiser (Surprise/Fear)
        # Normal ~ 0.06. Wide > 0.075.
        features["au_eye_wide"] = max(0.0, (avg_eye_open - 0.075) * 10)
        
        # AU6/7: Cheek Raiser / Lid Tightener (Happy/Anger)
        # Squint < 0.05
        features["au_eye_squint"] = max(0.0, (0.055 - avg_eye_open) * 10)

        # AU12: Lip Corner Puller (Happy)
        features["au_smile"] = max(0.0, (smile_curve - 0.01) * 10)

        # AU15: Lip Corner Depressor (Sad)
        features["au_frown"] = max(0.0, (-smile_curve - 0.01) * 10)

        # AU25/26/27: Mouth Open (Surprise/Fear/Happy)
        features["au_mouth_open"] = max(0.0, (mouth_open - 0.03) * 10)
        
        # AU20: Lip Stretch (Fear/Grimace)
        features["au_lip_stretch"] = max(0.0, (mouth_width - 0.45) * 5)

        # AU9: Nose Wrinkler (Disgust/Stressed)
        # Normal ~ 0.09. Scrunch < 0.07.
        features["au_nose_scrunch"] = max(0.0, (0.08 - nose_lip_dist) * 15)

        return features

    def _detect_emotion_rules(self, features):
        """
        Detect emotions based on Action Unit combinations (FACS-inspired).
        """
        scores = {e: 0.0 for e in EMOTIONS}
        
        # 1. Happy (Joy)
        # Primary: AU12 (Smile) + AU6 (Squint - Duchenne marker)
        # If eyes are wide, it reduces Happy score (likely Fear/Surprise).
        scores["Happy"] = features["au_smile"] * 1.5
        if features["au_eye_wide"] > 0.2:
            scores["Happy"] *= 0.3 # Penalty for wide eyes

        # 2. Sad (Sadness)
        # Primary: AU15 (Frown) + AU1 (Inner Brow Raise - difficult to separate, using general logic)
        # + AU4 (Brow Lowerer/Frown)
        # Improved: Added brow squeeze and raised threshold for frown to reduce false positives
        scores["Sad"] = features["au_frown"] * 1.8 + features["au_brow_low"] * 0.5 + features["au_brow_squeeze"] * 0.3

        # 3. Angry (Anger)
        # Primary: AU4 (Brow Lower/Squeeze) + AU5/7 (Wide or Squint) + AU23 (Lip Tight)
        # Tuned: Increased threshold for brow squeeze and reduced multiplier
        scores["Angry"] = (features["au_brow_squeeze"] * 1.5 + 
                           features["au_brow_low"] * 0.8 + 
                           features["au_eye_squint"] * 0.5)

        # 4. Fear (Fear)
        # Primary: AU1+2 (Brow Raise) + AU5 (Eye Wide) + AU20 (Lip Stretch) + AU25 (Mouth Open)
        # Tuned: Reduced eye wide sensitivity
        scores["Fear"] = (features["au_brow_raise"] * 0.8 + 
                          features["au_eye_wide"] * 0.8 + 
                          features["au_lip_stretch"] * 1.0 +
                          features["au_mouth_open"] * 0.3)
        
        # 5. Surprise (Surprise)
        # Primary: AU1+2 (Brow Raise) + AU5 (Eye Wide) + AU26 (Jaw Drop)
        # Distinguished from Fear by lack of Lip Stretch/Grimace
        scores["Surprise"] = (features["au_brow_raise"] * 0.8 + 
                             features["au_eye_wide"] * 1.0 + 
                             features["au_mouth_open"] * 0.8)
        
        # 6. Disgust (Disgust)
        # Primary: AU9 (Nose Scrunch)
        # Reduced sensitivity to avoid false positives (User Report)
        scores["Disgust"] = features["au_nose_scrunch"] * 1.5

        # Conflict Resolution & Normalization
        
        # Fear vs Surprise:
        # Fear usually has higher brow raise + lip stretch.
        if scores["Fear"] > scores["Surprise"]:
            scores["Surprise"] *= 0.5
        else:
            scores["Fear"] *= 0.5

        # Happy vs Fear (Grimace check):
        # If smile is high but eyes are very wide, it might be fear grimace.
        if features["au_smile"] > 0.5 and features["au_eye_wide"] > 0.5:
            scores["Fear"] += 0.5
            scores["Happy"] *= 0.5

        # Neutral Calculation
        # If all activations are low, it's Neutral.
        max_activation = max(scores.values()) if scores else 0
        if max_activation < 0.6: # Increased from 0.4 to capture more "resting" faces
            scores["Neutral"] = 1.0 # Stronger neutral baseline

        # Softmax-like normalization
        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] /= total
        else:
            scores["Neutral"] = 1.0

        return scores
        
    def _smooth_predictions(self, current_probs):
        """
        Apply temporal smoothing using a rolling average.
        """
        self.history.append(current_probs)
        
        # Calculate average over history window
        # Stack history: (N, 7) array
        history_array = np.array(self.history)
        avg_probs = np.mean(history_array, axis=0)
        
        return avg_probs
