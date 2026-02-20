import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import sys

# Define project_dir at the top
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add src to path
sys.path.append(os.path.join(project_dir, "backend", "src"))

from config.settings import EMOTIONS

# Define base directory
DATA_DIR = os.path.join(project_dir, "datasets", "Images")
OUTPUT_DIR = os.path.join(project_dir, "backend", "assets", "landmarks")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def extract_landmarks(image_path):
    """
    Extracts 478 landmarks from an image.
    Returns flattened array of x, y coordinates (478 * 2).
    Ignores Z for simplicity unless needed for depth.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        # Take the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract (x, y) for each landmark
        # We normalize by image dimensions? 
        # MediaPipe returns normalized coordinates [0, 1] relative to image size. Perfect.
        coords = []
        for lm in face_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z]) # Include Z for 3D geometry if needed

        return np.array(coords, dtype=np.float32)
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_dataset():
    X = []
    y = []
    
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory {DATA_DIR} not found.")
        return

    print(f"Scanning {DATA_DIR}...")
    
    # Depending on structure: Images/Train/Angry or Images/Angry?
    # Let's assume Images/Train/Angry and Images/Validation/Angry from earlier exploration
    # Or just iterate all subfolders recursively?
    # Let's try to stick to "train" folder first if it exists, or just root subfolders.
    
    target_dirs = []
    
    # Check standard train/val structure
    train_dir = os.path.join(DATA_DIR, "train")
    if os.path.exists(train_dir):
        target_dirs.append(train_dir)
        val_dir = os.path.join(DATA_DIR, "validation") # Or 'val'
        if os.path.exists(val_dir):
            target_dirs.append(val_dir)
        elif os.path.exists(os.path.join(DATA_DIR, "val")):
             target_dirs.append(os.path.join(DATA_DIR, "val"))
    else:
        # Flat structure
        target_dirs.append(DATA_DIR)
        
    print(f"Processing directories: {target_dirs}")
    
    processed_count = 0
    
    for d in target_dirs:
        for emotion in EMOTIONS:
            emotion_path = os.path.join(d, emotion) # e.g. .../train/Angry
            # Case insensitive check if folder names differ (e.g. angry vs Angry)
            if not os.path.exists(emotion_path):
                 # Try lowercase
                 emotion_path_lower = os.path.join(d, emotion.lower())
                 if os.path.exists(emotion_path_lower):
                     emotion_path = emotion_path_lower
                 else:
                     print(f"Warning: Folder for {emotion} not found in {d}")
                     continue
            
            print(f"Processing {emotion} in {d}...")
            files = os.listdir(emotion_path)
            
            for file_name in tqdm(files, desc=f"{emotion}"):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(emotion_path, file_name)
                    landmarks = extract_landmarks(file_path)
                    
                    if landmarks is not None:
                        X.append(landmarks)
                        y.append(EMOTIONS.index(emotion))
                        processed_count += 1
                        
    if processed_count == 0:
        print("No faces detected or files found. Check paths.")
        return

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Extraction Complete. Shape: X={X.shape}, y={y.shape}")
    
    # Save
    np.save(os.path.join(OUTPUT_DIR, "X_landmarks.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_landmarks.npy"), y)
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_dataset()
