
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'mers', 'src'))

from visual_engine import VisualEngine

def test_emotion_logic_new():
    ve = VisualEngine()
    
    # Define test cases based on Action Unit (AU) values (0.0 to 1.0+)
    test_cases = [
        {
            "name": "Neutral Baseline",
            "features": {
                "au_smile": 0.0, "au_frown": 0.0,
                "au_brow_raise": 0.0, "au_brow_squeeze": 0.0, "au_brow_low": 0.0,
                "au_eye_wide": 0.0, "au_eye_squint": 0.0,
                "au_mouth_open": 0.0, "au_lip_stretch": 0.0,
                "au_nose_scrunch": 0.0
            },
            "expected": "Neutral"
        },
        {
            "name": "Happy (Duchenne)",
            "features": {
                "au_smile": 0.8, "au_frown": 0.0,
                "au_brow_raise": 0.0, "au_brow_squeeze": 0.0, "au_brow_low": 0.0,
                "au_eye_wide": 0.0, "au_eye_squint": 0.3, # Squinting eyes
                "au_mouth_open": 0.1, "au_lip_stretch": 0.0,
                "au_nose_scrunch": 0.0
            },
            "expected": "Happy"
        },
        {
            "name": "Sadness",
            "features": {
                "au_smile": 0.0, "au_frown": 0.6,
                "au_brow_raise": 0.0, "au_brow_squeeze": 0.0, "au_brow_low": 0.4,
                "au_eye_wide": 0.0, "au_eye_squint": 0.0,
                "au_mouth_open": 0.0, "au_lip_stretch": 0.0,
                "au_nose_scrunch": 0.0
            },
            "expected": "Sad"
        },
        {
            "name": "Anger",
            "features": {
                "au_smile": 0.0, "au_frown": 0.0,
                "au_brow_raise": 0.0, "au_brow_squeeze": 0.8, "au_brow_low": 0.6,
                "au_eye_wide": 0.0, "au_eye_squint": 0.4, # Glare
                "au_mouth_open": 0.0, "au_lip_stretch": 0.0,
                "au_nose_scrunch": 0.0
            },
            "expected": "Angry"
        },
        {
            "name": "Fear",
            "features": {
                "au_smile": 0.1, "au_frown": 0.0,
                "au_brow_raise": 0.8, "au_brow_squeeze": 0.2, "au_brow_low": 0.0,
                "au_eye_wide": 0.9, "au_eye_squint": 0.0,
                "au_mouth_open": 0.4, "au_lip_stretch": 0.7, # Grimace
                "au_nose_scrunch": 0.0
            },
            "expected": "Fear"
        },
        {
            "name": "Focused (Surprise)",
            "features": {
                "au_smile": 0.0, "au_frown": 0.0,
                "au_brow_raise": 0.7, "au_brow_squeeze": 0.0, "au_brow_low": 0.0,
                "au_eye_wide": 0.8, "au_eye_squint": 0.0,
                "au_mouth_open": 0.6, "au_lip_stretch": 0.0, # No grimace
                "au_nose_scrunch": 0.0
            },
            "expected": "Focused"
        },
        {
            "name": "Stressed (Disgust)",
            "features": {
                "au_smile": 0.0, "au_frown": 0.0,
                "au_brow_raise": 0.0, "au_brow_squeeze": 0.0, "au_brow_low": 0.3,
                "au_eye_wide": 0.0, "au_eye_squint": 0.2,
                "au_mouth_open": 0.0, "au_lip_stretch": 0.0,
                "au_nose_scrunch": 0.8 # Strong scrunch
            },
            "expected": "Stressed"
        },
        {
            "name": "Fear Grimace (Previously Happy?)",
            "features": {
                "au_smile": 0.6, # Looks like smile
                "au_frown": 0.0,
                "au_brow_raise": 0.0, "au_brow_squeeze": 0.0, "au_brow_low": 0.0,
                "au_eye_wide": 0.8, # But eyes wide!
                "au_eye_squint": 0.0,
                "au_mouth_open": 0.3, "au_lip_stretch": 0.5,
                "au_nose_scrunch": 0.0
            },
            "expected": "Fear" # Should be Fear due to wide eyes + grimace logic
        }
    ]

    print(f"{'Case':<35} | {'Expected':<10} | {'Predicted':<10} | {'Status':<6} | {'Scores'}")
    print("-" * 120)

    for case in test_cases:
        # Fill missing AUs with 0.0
        full_features = {
            "au_smile": 0.0, "au_frown": 0.0,
            "au_brow_raise": 0.0, "au_brow_squeeze": 0.0, "au_brow_low": 0.0,
            "au_eye_wide": 0.0, "au_eye_squint": 0.0,
            "au_mouth_open": 0.0, "au_lip_stretch": 0.0,
            "au_nose_scrunch": 0.0
        }
        full_features.update(case["features"])

        scores = ve._detect_emotion_rules(full_features)
        predicted = max(scores, key=scores.get)
        status = "PASS" if predicted == case["expected"] else "FAIL"
        
        sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        top_3 = list(sorted_scores.items())[:3]
        scores_str = ", ".join([f"{k}:{v:.2f}" for k,v in top_3])

        print(f"{case['name']:<35} | {case['expected']:<10} | {predicted:<10} | {status:<6} | {scores_str}")

if __name__ == "__main__":
    test_emotion_logic_new()
