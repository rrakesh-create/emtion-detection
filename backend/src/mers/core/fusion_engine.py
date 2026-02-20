import datetime
import numpy as np

from config.settings import (
    MODEL_EMOTIONS,
    EMOTIONS,
    EMOTION_MAPPING,
    EMOTION_COLORS,
    get_color
)
from mers.core.xai_engine import XAIEngine

class FusionEngine:
    def __init__(self):
        self.xai = XAIEngine()
        self.override_weights = None # {"visual": 0.5, "audio": 0.5}
        self.disabled_modalities = set() # {"visual", "audio"}

    def set_config(self, weights=None, disabled=None):
        """
        Runtime configuration for fusion.
        weights: dict {"visual": float, "audio": float}
        disabled: list ["visual", "audio"]
        """
        if weights: 
            self.override_weights = weights
        else:
            self.override_weights = None # Reset if None passed
            
        if disabled is not None: 
            self.disabled_modalities = set(disabled)

    def fuse(self, visual_probs, audio_probs, visual_features=None, audio_features=None):
        """
        Adaptive fusion based on confidence + XAI Generation.
        Updated to return new JSON schema structure with output mapping.
        """
        if visual_features is None:
            visual_features = {}
        if audio_features is None:
            audio_features = {}

        # 0. Apply Disabling Logic
        if "visual" in self.disabled_modalities:
            visual_probs = None
        if "audio" in self.disabled_modalities:
            audio_probs = None

        # 1. Determine Weights & Modalities
        w_v, w_a = 0.0, 0.0
        modalities = []

        v_conf = np.max(visual_probs) if visual_probs is not None else 0.0
        a_conf = np.max(audio_probs) if audio_probs is not None else 0.0

        fused_probs = None

        if visual_probs is None and audio_probs is None:
            # All Missing
            return self._build_empty_response("Insufficient audio/visual clarity")

        if visual_probs is None:
            # Audio Only
            fused_probs = audio_probs
            w_v, w_a = 0.0, 1.0
            modalities = ["audio"]

        elif audio_probs is None:
            # Visual Only
            fused_probs = visual_probs
            w_v, w_a = 1.0, 0.0
            modalities = ["visual"]

        else:
            # Multimodal Fusion
            modalities = ["visual", "audio"]

            if self.override_weights:
                # Use User Defined Weights
                w_v = self.override_weights.get("visual", 0.5)
                w_a = self.override_weights.get("audio", 0.5)
                
                # Normalize just in case
                total = w_v + w_a
                if total > 0:
                    w_v /= total
                    w_a /= total
                else:
                    w_v, w_a = 0.5, 0.5
            else:
                # Dynamic Weighting based on Confidence
                # We want to trust the modality that is more confident, 
                # but not completely discard the other unless it's very weak.
                
                # 1. Normalize confidences to weights
                total_conf = v_conf + a_conf
                if total_conf > 0:
                    w_v = v_conf / total_conf
                    w_a = a_conf / total_conf
                else:
                    w_v, w_a = 0.5, 0.5

                # 2. Heuristic: Audio Reliability Check
                # If audio predicts 'Angry' or 'Fear' but energy is low (background noise), reduce its weight.
                audio_energy = audio_features.get("energy_mean", 0.0)
                # Check if Audio top emotion is Angry/Fear
                # audio_probs is in EMOTIONS order
                a_top_idx = np.argmax(audio_probs)
                a_top_emo = EMOTIONS[a_top_idx]
                
                if a_top_emo in ["Angry", "Fear"] and audio_energy < 0.02:
                     # Low energy "Angry" is usually noise.
                     w_a *= 0.2
                     w_v = 1.0 - w_a

                # 3. Heuristic: Visual Reliability Check
                # If visual is extremely unsure (<0.3), trust it less, but don't nuke it.
                if v_conf < 0.3:
                    w_v *= 0.5
                    # Renormalize
                    total = w_v + w_a
                    w_v /= total
                    w_a /= total

            fused_probs = (w_v * visual_probs) + (w_a * audio_probs)

        # 2. Get Prediction (Model Space)
        final_idx = np.argmax(fused_probs)
        raw_emotion = EMOTIONS[final_idx]
        confidence = float(fused_probs[final_idx])

        # 3. Map to Output Space (User Requirements)
        output_emotion = EMOTION_MAPPING.get(raw_emotion, "Neutral")

        # 4. Generate Explanation (XAI)
        # We pass the Output Emotion so XAI generates relevant text
        # But we pass raw features
        explanation_dict = self.xai.explain(
            {"emotion": output_emotion, "modalities_used": modalities},
            visual_features,
            audio_features
        )

        # 5. Build Final JSON Response
        # {
        #   "emotion": "Focused",
        #   "confidence": 0.81,
        #   "color": "#1ABC9C",
        #   "modality_weights": { "audio": 0.65, "visual": 0.35 },
        #   "explanation": { "visual": "...", "audio": "..." },
        #   "timestamp": "..."
        # }

        color_hex = get_color(output_emotion, format="hex")

        response = {
            "emotion": output_emotion,
            "confidence": round(confidence, 2),
            "color": color_hex,
            "modality_weights": {
                "visual": round(w_v, 2),
                "audio": round(w_a, 2)
            },
            "explanation": explanation_dict,
            "timestamp": datetime.datetime.now().isoformat(),
            "debug": {
                "visual_confidence": round(v_conf, 2),
                "audio_confidence": round(a_conf, 2)
            }
        }
        return response

    def _build_empty_response(self, reason):
        return {
            "emotion": "Neutral",
            "confidence": 0.0,
            "color": "#95A5A6",
            "modality_weights": {"visual": 0.0, "audio": 0.0},
            "explanation": {"error": reason},
            "timestamp": datetime.datetime.now().isoformat()
        }
