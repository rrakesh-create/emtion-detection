
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'mers', 'src'))

from visual_engine import VisualEngine

def test_emotion_rules():
    ve = VisualEngine()
    
    # Define test cases based on typical feature values
    test_cases = [
        {
            "name": "Fear Grimace (User: Fear->Happy)",
            "features": {
                "mouth_open": 0.08,
                "mouth_width": 0.55,   # Stretched mouth
                "eye_open": 0.09,      # Wide eyes (Fear trait)
                "eyebrow_raise": 0.12, # Raised
                "brow_inner_dist": 0.2,
                "nose_lip_dist": 0.1,
                "smile_ratio": 0.04    # Grimace looks like smile > 0.03
            },
            "expected": "Fear"
        },
        {
            "name": "Weak Surprise (User: Surprise->Neutral)",
            "features": {
                "mouth_open": 0.045,   # Below old 0.05 threshold
                "mouth_width": 0.3,
                "eye_open": 0.075,     # Below old 0.08 threshold
                "eyebrow_raise": 0.09,
                "brow_inner_dist": 0.2,
                "nose_lip_dist": 0.1,
                "smile_ratio": 0.0
            },
            "expected": "Focused"
        },
        {
            "name": "High-Brow Neutral (User: Neutral->Fear)",
            "features": {
                "mouth_open": 0.02,
                "mouth_width": 0.4,
                "eye_open": 0.075,     # Naturally wide?
                "eyebrow_raise": 0.105,# Naturally high?
                "brow_inner_dist": 0.2,
                "nose_lip_dist": 0.1,
                "smile_ratio": 0.0
            },
            "expected": "Neutral"
        },
        {
            "name": "Sad Frown (User: Sad->Angry)",
            "features": {
                "mouth_open": 0.02,
                "mouth_width": 0.4,
                "eye_open": 0.05,
                "eyebrow_raise": 0.05, # Low brows (frown)
                "brow_inner_dist": 0.16, # Not fully squeezed (< 0.15 is Angry)
                "nose_lip_dist": 0.1,
                "smile_ratio": -0.04   # Sad mouth
            },
            "expected": "Sad"
        },
        {
            "name": "Angry (User: Angry->Fear?)",
            "features": {
                "mouth_open": 0.01,
                "mouth_width": 0.4,
                "eye_open": 0.04,
                "eyebrow_raise": 0.05, # Low
                "brow_inner_dist": 0.12, # Squeezed
                "nose_lip_dist": 0.1,
                "smile_ratio": 0.0
            },
            "expected": "Angry"
        }
    ]

    print(f"{'Case':<35} | {'Expected':<10} | {'Predicted':<10} | {'Scores'}")
    print("-" * 100)

    for case in test_cases:
        scores = ve._detect_emotion_rules(case["features"])
        # Find max score
        predicted = max(scores, key=scores.get)
        
        # Sort scores for display
        sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        top_3 = list(sorted_scores.items())[:3]
        scores_str = ", ".join([f"{k}:{v:.2f}" for k,v in top_3])

        print(f"{case['name']:<35} | {case['expected']:<10} | {predicted:<10} | {scores_str}")

if __name__ == "__main__":
    test_emotion_rules()
