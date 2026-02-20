
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, "backend", "src"))

from core.visual_engine import VisualEngine
from config.settings import EMOTIONS, EMOTION_MAPPING

# Configuration
DATASET_DIR = os.path.join(project_dir, "datasets", "FER2013", "test")
OUTPUT_DIR = os.path.join(project_dir, "evaluation_results", "benchmark_visual_logic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FER2013 Labels -> MERS Labels
# MERS EMOTIONS: ["Happy", "Sad", "Angry", "Fear", "Neutral", "Focused", "Stressed"]
# FER2013: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

FER_TO_MERS = {
    'angry': 'Angry',
    'disgust': 'Stressed', # Mapped Disgust -> Stressed
    'fear': 'Fear',
    'happy': 'Happy',
    'neutral': 'Neutral',
    'sad': 'Sad',
    'surprise': 'Focused' # Mapped Surprise -> Focused
}

def benchmark():
    print("Loading VisualEngine (Rule-Based)...")
    try:
        engine = VisualEngine()
    except Exception as e:
        print(f"Failed to load engine: {e}")
        return

    y_true = []
    y_pred = []
    
    # Iterate through FER classes
    classes = os.listdir(DATASET_DIR)
    print(f"Found classes: {classes}")

    for cls in classes:
        if cls not in FER_TO_MERS:
            continue
            
        mers_label = FER_TO_MERS[cls]
        cls_dir = os.path.join(DATASET_DIR, cls)
        
        images = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.png'))]
        print(f"Processing {cls} -> {mers_label} ({len(images)} images)...")
        
        # Limit for speed if needed, but let's do all
        # images = images[:100] 
        
        for img_name in tqdm(images):
            img_path = os.path.join(cls_dir, img_name)
            
            # Load and Preprocess
            # FER is 48x48 Grayscale. MediaPipe needs RGB and reasonable size.
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Resize to simulate webcam frame (e.g. 640x480)
            # Upscaling 48x48 might be blurry but MediaPipe needs pixels
            img = cv2.resize(img, (480, 480)) 
            
            # Process
            probs, _, _, _ = engine.process_frame(img)
            
            if probs is not None:
                # Get predicted class index
                pred_idx = np.argmax(probs)
                pred_emotion = EMOTIONS[pred_idx]
                
                y_true.append(mers_label)
                y_pred.append(pred_emotion)
            else:
                # Face not detected
                pass

    # Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=sorted(list(set(y_true + y_pred))))
    print(report)
    
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
        
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title("Confusion Matrix: Rule-Based Logic vs FER2013")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    benchmark()
