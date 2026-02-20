import cv2
import os
import glob
import sys
import numpy as np
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
from PIL import Image

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, "backend", "src"))
from config.settings import EMOTIONS, INPUT_IMAGE_SIZE

def get_emotion_from_filename(filename, dataset_type="generic"):
    """
    Parses filename to extract emotion based on dataset conventions.
    Returns: Emotion string (e.g., "Happy") or None.
    """
    filename = os.path.basename(filename)
    name, ext = os.path.splitext(filename)
    
    # MERS Emotions: ["Angry", "Happy", "Sad", "Neutral", "Fear", "Surprise", "Disgust"]
    
    if dataset_type == "ravdess":
        # 02-01-06-01-02-01-12.mp4
        # 3rd item: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=surprise
        parts = name.split('-')
        if len(parts) >= 3:
            code = parts[2]
            mapping = {
                "01": "Neutral", "02": "Neutral", "03": "Happy",
                "04": "Sad", "05": "Angry", "06": "Fear",
                "07": "Disgust", "08": "Surprise"
            }
            return mapping.get(code)

    elif dataset_type == "crema":
        # 1001_DFA_ANG_XX.flv
        parts = name.split('_')
        if len(parts) >= 3:
            code = parts[2]
            mapping = {
                "ANG": "Angry", "DIS": "Disgust", "FEA": "Fear",
                "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"
            }
            return mapping.get(code)
            
    # Add other dataset logic here (Savee usually audio-only, but has video too)
    
    return None

def extract_frames(video_root, output_root, target_fps=2):
    """
    Extracts frames from videos, crops faces, and saves them sorted by emotion.
    target_fps: Extract 1 frame every (video_fps / target_fps) frames.
    """
    print(f"Starting Frame Extraction from {video_root} to {output_root}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for Face Detection")
    
    mtcnn = MTCNN(image_size=INPUT_IMAGE_SIZE, margin=20, keep_all=False, select_largest=True, device=device)
    
    # Create output directories
    for emo in EMOTIONS:
        os.makedirs(os.path.join(output_root, emo), exist_ok=True)
        
    # Walk through video root
    video_extensions = ['*.mp4', '*.avi', '*.flv', '*.mov', '*.mkv']
    files = []
    for ext in video_extensions:
        files.extend(glob.glob(os.path.join(video_root, "**", ext), recursive=True))
        
    print(f"Found {len(files)} video files.")
    
    processed_count = 0
    faces_saved = 0
    
    for vid_path in tqdm(files):
        # Determine Dataset Type heuristic
        fname = os.path.basename(vid_path)
        dtype = "generic"
        if len(fname.split('-')) >= 6: dtype = "ravdess"
        elif len(fname.split('_')) >= 3: dtype = "crema"
        
        emotion = get_emotion_from_filename(vid_path, dtype)
        if not emotion or emotion not in EMOTIONS:
            continue
            
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        
        step = int(fps / target_fps)
        if step < 1: step = 1
        
        frame_idx = 0
        saved_from_vid = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % step == 0:
                # Detect and Crop
                try:
                    # Convert to RGB PIL
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img)
                    
                    # Get cropped face (returns tensor or None)
                    # save_path needs to be a file path
                    save_name = f"{os.path.splitext(fname)[0]}_f{frame_idx}.jpg"
                    save_path = os.path.join(output_root, emotion, save_name)
                    
                    # MTCNN save function is convenient
                    # But we want to resize to INPUT_IMAGE_SIZE (224)
                    # mtcnn(img, save_path=...) crops and saves.
                    
                    # Check if face exists first to avoid empty files?
                    # mtcnn forward returns cropped tensor.
                    face_tensor = mtcnn(img_pil, save_path=save_path)
                    
                    if face_tensor is not None:
                        faces_saved += 1
                        saved_from_vid += 1
                        
                except Exception as e:
                    # print(f"Error processing frame: {e}")
                    pass
            
            frame_idx += 1
            
        cap.release()
        processed_count += 1
        
    print(f"Extraction Complete.")
    print(f"Processed {processed_count} videos.")
    print(f"Saved {faces_saved} face images to {output_root}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_frames.py <path_to_videos_root> <path_to_output_visual_data>")
        print(f"Example: python extract_frames.py {os.path.join('datasets', 'Ravdess_Video')} {os.path.join('backend', 'assets', 'visual_data')}")
    else:
        extract_frames(sys.argv[1], sys.argv[2])
