
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
        # --- ORIGINAL NORMAL CASES ---
        {
            "name": "Normal Neutral",
            "features": {
                "mouth_open": 0.02, "mouth_width": 0.4, "eye_open": 0.06,
                "eyebrow_raise": 0.08, "brow_inner_dist": 0.2, "nose_lip_dist": 0.1,
                "smile_ratio": 0.0
            },
            "expected": "Neutral"
        },
        {
            "name": "Normal Happy",
            "features": {
                "mouth_open": 0.05, "mouth_width": 0.5, "eye_open": 0.05,
                "eyebrow_raise": 0.08, "brow_inner_dist": 0.2, "nose_lip_dist": 0.1,
                "smile_ratio": 0.05
            },
            "expected": "Happy"
        },
        {
            "name": "Normal Sad",
            "features": {
                "mouth_open": 0.02, "mouth_width": 0.4, "eye_open": 0.05,
                "eyebrow_raise": 0.07, "brow_inner_dist": 0.18, "nose_lip_dist": 0.1,
                "smile_ratio": -0.04
            },
            "expected": "Sad"
        },
        
        # --- PROBLEMATIC CASES (FIXED) ---
        {
            "name": "Fear Grimace (User: Fear->Happy)",
            "features": {
                "mouth_open": 0.08,
                "mouth_width": 0.55, "eye_open": 0.09,
                "eyebrow_raise": 0.12, "brow_inner_dist": 0.2, "nose_lip_dist": 0.1,
                "smile_ratio": 0.04 # Grimace
            },
            "expected": "Fear"
        },
        {
            "name": "Weak Surprise (User: Surprise->Neutral)",
            "features": {
                "mouth_open": 0.045, "mouth_width": 0.3, "eye_open": 0.075,
                "eyebrow_raise": 0.09, "brow_inner_dist": 0.2, "nose_lip_dist": 0.1,
                "smile_ratio": 0.0
            },
            "expected": "Focused"
        },
        {
            "name": "High-Brow Neutral (User: Neutral->Fear)",
            "features": {
                "mouth_open": 0.02, "mouth_width": 0.4, "eye_open": 0.075,
                "eyebrow_raise": 0.105, "brow_inner_dist": 0.2, "nose_lip_dist": 0.1,
                "smile_ratio": 0.0
            },
            "expected": "Neutral"
        },
        {
            "name": "Sad Frown (User: Sad->Angry)",
            "features": {
                "mouth_open": 0.02, "mouth_width": 0.4, "eye_open": 0.05,
                "eyebrow_raise": 0.05, "brow_inner_dist": 0.16, "nose_lip_dist": 0.1,
                "smile_ratio": -0.04
            },
            "expected": "Sad"
        },
        {
            "name": "Angry (User: Angry->Fear?)",
            "features": {
                "mouth_open": 0.01, "mouth_width": 0.4, "eye_open": 0.04,
                "eyebrow_raise": 0.05, "brow_inner_dist": 0.12, "nose_lip_dist": 0.1,
                "smile_ratio": 0.0
            },
            "expected": "Angry"
        }
    ]

    print(f"{'Case':<40} | {'Expected':<10} | {'Predicted':<10} | {'Status':<6} | {'Scores'}")
    print("-" * 120)

    for case in test_cases:
        scores = ve._detect_emotion_rules(case["features"])
        predicted = max(scores, key=scores.get)
        status = "PASS" if predicted == case["expected"] else "FAIL"
        
        sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        top_3 = list(sorted_scores.items())[:3]
        scores_str = ", ".join([f"{k}:{v:.2f}" for k,v in top_3])

        print(f"{case['name']:<40} | {case['expected']:<10} | {predicted:<10} | {status:<6} | {scores_str}")

if __name__ == "__main__":
    test_emotion_rules()
