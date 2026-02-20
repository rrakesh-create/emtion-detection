
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config.settings import EMOTIONS, DEVICE, MODELS_DIR
from mers.core.visual_engine import VisualEngine

class VisualEngineCNN(VisualEngine):
    """
    Visual Engine that uses MediaPipe for Face Detection/Landmarks (for XAI)
    and a fine-tuned EfficientNet-B0 for Emotion Classification.
    """
    def __init__(self, model_path=None):
        super().__init__() # Initialize MediaPipe from parent
        
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "visual_efficientnet.pth")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VisualEngineCNN] Loading model from {model_path} on {self.device}...")
        
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model.to(self.device)
        
        # Preprocessing matching training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # FER2013 Class Order (must match training script)
        self.model_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        self.idx_to_class = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Neutral',
            5: 'Sad',
            6: 'Surprise'
        }

    def _load_model(self, path):
        # Reconstruct model architecture
        model = models.efficientnet_b0(weights=None) # No weights needed, we load state_dict
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 7) # 7 classes
        
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)
                print("[VisualEngineCNN] Model loaded successfully.")
            except Exception as e:
                print(f"[VisualEngineCNN] Error loading weights: {e}")
                print("[VisualEngineCNN] WARNING: Using random weights!")
        else:
            print(f"[VisualEngineCNN] Model file not found at {path}. Using random weights.")
            
        return model

    def process_frame(self, frame):
        # 1. Get Landmarks & Box using parent (MediaPipe)
        # We need the box to crop the face for the CNN
        # Parent returns: prob_array, face_box, features, landmarks
        # We will ignore parent's prob_array
        
        # Run detection manually to get the box first, or reuse parent logic?
        # Parent logic couples detection and rule-based inference.
        # Let's override process_frame but reuse internal helpers if possible.
        # Actually, parent process_frame calls _detect_emotion_rules at the end.
        # It's better to copy-paste the structure or call super and overwrite probs.
        
        # Calling super is easiest to get landmarks/features
        rule_probs, face_box, features, landmarks = super().process_frame(frame)
        
        if face_box is None:
            return None, None, {}, None

        # 2. Crop Face
        x1, y1, x2, y2 = face_box
        h, w, _ = frame.shape
        
        # Add padding
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return rule_probs, face_box, features, landmarks

        # 3. CNN Inference
        try:
            # Convert BGR to RGB (PIL)
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
            # Map CNN probabilities to MERS EMOTIONS order
            # MERS: ["Happy", "Sad", "Angry", "Fear", "Neutral", "Surprise", "Disgust"]
            # CNN (FER): 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Neutral, 5=Sad, 6=Surprise
            
            cnn_probs = np.zeros(len(EMOTIONS))
            
            # Map
            # Happy (Index 0 in MERS) <- CNN Index 3
            cnn_probs[0] = probs[3] 
            # Sad (Index 1 in MERS) <- CNN Index 5
            cnn_probs[1] = probs[5]
            # Angry (Index 2 in MERS) <- CNN Index 0
            cnn_probs[2] = probs[0]
            # Fear (Index 3 in MERS) <- CNN Index 2
            cnn_probs[3] = probs[2]
            # Neutral (Index 4 in MERS) <- CNN Index 4
            cnn_probs[4] = probs[4]
            # Surprise (Index 5 in MERS) <- CNN Index 6
            cnn_probs[5] = probs[6]
            # Disgust (Index 6 in MERS) <- CNN Index 1
            cnn_probs[6] = probs[1]
            
            # 4. Temporal Smoothing
            # VisualEngineCNN inherits _smooth_predictions from VisualEngine
            smooth_probs = self._smooth_predictions(cnn_probs)
            
            return smooth_probs, face_box, features, landmarks
            
        except Exception as e:
            print(f"[VisualEngineCNN] Inference Error: {e}")
            return rule_probs, face_box, features, landmarks
