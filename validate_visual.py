import os
import argparse
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from mers.src.visual_engine import VisualEngine
from mers.config import EMOTIONS, EMOTION_MAPPING, EMOTION_EMOJIS

# Reverse mapping for evaluation
# We need to map dataset labels (often standard 7) to our OUTPUT_EMOTIONS
# Standard: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
# MERS Output: Angry, Stressed, Fear, Happy, Sad, Focused, Neutral
STANDARD_TO_MERS = {
    "Angry": "Angry",
    "Disgust": "Stressed",
    "Fear": "Fear",
    "Happy": "Happy",
    "Sad": "Sad",
    "Surprise": "Focused",
    "Neutral": "Neutral"
}

def load_dataset(dataset_path):
    """
    Loads images from a directory structure:
    dataset_path/
      Angry/
        img1.jpg
      Happy/
        ...
    """
    images = []
    labels = []
    
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset path not found: {dataset_path}")
        return [], []

    print(f"[Info] Scanning dataset at {dataset_path}...")
    
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            # Normalize label
            raw_label = folder.capitalize()
            if raw_label in STANDARD_TO_MERS:
                target_label = STANDARD_TO_MERS[raw_label]
                
                for file in os.listdir(folder_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(folder_path, file)
                        images.append(img_path)
                        labels.append(target_label)
            else:
                print(f"[Warning] Skipping unknown class folder: {folder}")

    print(f"[Info] Found {len(images)} images across {len(set(labels))} classes.")
    return images, labels

def evaluate_engine(engine, images, labels):
    y_true = []
    y_pred = []
    
    print("[Info] Starting Inference...")
    for i, (img_path, label) in enumerate(zip(images, labels)):
        if i % 50 == 0:
            print(f"Processing {i}/{len(images)}...")
            
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        # Inference
        probs, _, _, _ = engine.process_frame(frame)
        
        if probs is None:
            # Face not detected
            pred = "Neutral" # Fallback
        else:
            idx = np.argmax(probs)
            pred = EMOTIONS[idx]
            
        y_true.append(label)
        y_pred.append(pred)
        
    return y_true, y_pred

def generate_report(y_true, y_pred, output_dir="evaluation_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Classification Report
    report = classification_report(y_true, y_pred, target_names=EMOTIONS, labels=EMOTIONS, zero_division=0)
    print("\n--- Classification Report ---")
    print(report)
    
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=EMOTIONS, yticklabels=EMOTIONS, cmap='Blues')
    plt.title("Confusion Matrix (Visual Engine)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    print(f"[Info] Confusion Matrix saved to {output_dir}/confusion_matrix.png")
    
    # 3. Emoji Accuracy
    print("\n--- Emoji Correlation Check ---")
    correct = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1
    
    acc = correct / len(y_true) if y_true else 0
    print(f"Emoji Display Accuracy: {acc*100:.2f}%")
    if acc < 0.85:
        print("[FAIL] Accuracy below 85% threshold. Migration recommended.")
    else:
        print("[PASS] Accuracy meets threshold.")

def main():
    parser = argparse.ArgumentParser(description="MERS Visual Validation Tool")
    parser.add_argument("--dataset", type=str, help="Path to evaluation dataset (FER2013/AffectNet format)")
    args = parser.parse_args()
    
    if not args.dataset:
        print("Usage: python validate_visual.py --dataset /path/to/test_data")
        print("No dataset provided. Creating a dummy report for demonstration.")
        # Dummy data
        y_true = ["Happy"] * 50 + ["Sad"] * 50 + ["Angry"] * 50
        y_pred = ["Happy"] * 40 + ["Neutral"] * 10 + ["Sad"] * 30 + ["Neutral"] * 20 + ["Angry"] * 45 + ["Happy"] * 5
        generate_report(y_true, y_pred)
        return

    # Load Engine
    try:
        engine = VisualEngine()
    except Exception as e:
        print(f"Failed to load VisualEngine: {e}")
        return

    # Load Data
    images, labels = load_dataset(args.dataset)
    if not images:
        return

    # Evaluate
    y_true, y_pred = evaluate_engine(engine, images, labels)
    
    # Report
    generate_report(y_true, y_pred)

if __name__ == "__main__":
    main()
