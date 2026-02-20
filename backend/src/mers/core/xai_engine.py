from typing import Dict, Any

class XAIEngine:
    """
    Explainable AI Engine for MERS.
    Generates human-readable explanations for emotion predictions
    based on visual (geometric) and audio (prosodic) features.
    Updated to return structured dictionary for UI consumption.
    """

    def __init__(self):
        pass

    def explain(self,
                fusion_result: Dict[str, Any],
                visual_features: Dict[str, Any],
                audio_features: Dict[str, Any]) -> Dict[str, str]:
        """
        Constructs a structured explanation dictionary.
        Returns:
            {
                "visual": "...",
                "audio": "..."
            }
        """
        emotion = fusion_result.get("emotion", "Neutral")
        modalities = fusion_result.get("modalities_used", [])

        explanation = {
            "visual": "No visual cues detected.",
            "audio": "No audio cues detected."
        }

        # 1. Visual Explanation
        if "visual" in modalities and visual_features:
            explanation["visual"] = self._explain_visual(emotion, visual_features)
        elif "visual" not in modalities:
            explanation["visual"] = "Camera inactive or face not detected."

        # 2. Audio Explanation
        if "audio" in modalities and audio_features:
            explanation["audio"] = self._explain_audio(emotion, audio_features)
        elif "audio" not in modalities:
            explanation["audio"] = "Microphone inactive or silence detected."

        return explanation

    # Backwards compatibility alias if needed
    def generate_explanation(self, emotion, visual_features, audio_features, modalities_used):
        # This is just a wrapper to match the old signature if called directly
        # But we prefer using explain() with the full context
        return self.explain({"emotion": emotion, "modalities_used": modalities_used}, visual_features, audio_features)

    def _explain_visual(self, emotion: str, features: Dict[str, Any]) -> str:
        """
        Interprets geometric features like 'mouth_open', 'eyebrow_raise', etc.
        """
        reasons = []

        # Extract features
        mouth_open = features.get("mouth_open", 0)
        smile_ratio = features.get("smile_ratio", 0) # Width/Height or corner dist
        eyebrow_raise = features.get("eyebrow_raise", 0)

        # Rules for Output Emotions
        if emotion == "Happy":
            if smile_ratio > 0.02:
                reasons.append("lip corners pulled up")
            if eyebrow_raise > 0.02:
                reasons.append("relaxed eyebrows")

        elif emotion == "Sad":
            if smile_ratio < -0.01:
                reasons.append("lip corners drooping")
            if eyebrow_raise < 0.05:
                reasons.append("inner eyebrows raised")

        elif emotion == "Angry":
            if eyebrow_raise < 0.05:
                reasons.append("eyebrows furrowed")
            if mouth_open < 0.02:
                reasons.append("lips pressed/tension")

        elif emotion == "Fear":
            if mouth_open > 0.02:
                reasons.append("mouth slightly open")
            if eyebrow_raise > 0.08:
                reasons.append("eyebrows raised high")

        elif emotion == "Focused": # Mapped from Surprise/Neutral
            if mouth_open < 0.02:
                reasons.append("mouth closed")
            if 0.02 < eyebrow_raise < 0.08:
                reasons.append("alert gaze")
            reasons.append("steady head position")

        elif emotion == "Stressed": # Mapped from Disgust/Fear
            if eyebrow_raise < 0.05:
                reasons.append("eyebrow tension")
            reasons.append("facial tightness")

        elif emotion == "Neutral":
            reasons.append("relaxed facial features")
            reasons.append("no significant muscle activation")

        if not reasons:
            return f"Facial landmarks align with {emotion.lower()} expression patterns."

        return f"{', '.join(reasons)}."

    def _explain_audio(self, emotion: str, features: Dict[str, Any]) -> str:
        """
        Interprets prosodic features like pitch, energy, rate.
        """
        reasons = []

        # Features come from AudioEngine._extract_prosody
        pitch = features.get("pitch_mean", 0)
        energy = features.get("energy_mean", 0)
        rate = features.get("speaking_rate", 0)

        # Rules for Output Emotions
        if emotion == "Happy":
            if pitch > 150:
                reasons.append("higher pitch")
            if energy > 0.05:
                reasons.append("upbeat energy")

        elif emotion == "Sad":
            if pitch < 120:
                reasons.append("lower pitch")
            if energy < 0.02:
                reasons.append("low vocal energy")
            if rate < 3.0:
                reasons.append("slower speaking rate")

        elif emotion == "Angry":
            if energy > 0.1:
                reasons.append("high vocal energy")
            if rate > 4.0:
                reasons.append("rapid speech")
            reasons.append("abrupt tonal shifts")

        elif emotion == "Fear":
            if pitch > 200:
                reasons.append("high pitch")
            reasons.append("tremulous quality")

        elif emotion == "Focused":
            if 3.0 < rate < 5.0:
                reasons.append("steady speaking rate")
            reasons.append("consistent pitch contour")

        elif emotion == "Stressed":
            if rate > 4.5:
                reasons.append("rapid speaking rate")
            if energy > 0.08:
                reasons.append("tense vocal quality")

        elif emotion == "Neutral":
            reasons.append("steady pitch")
            reasons.append("moderate volume")

        if not reasons:
            return f"Vocal prosody matches {emotion.lower()} profile."

        return f"{', '.join(reasons)}."
