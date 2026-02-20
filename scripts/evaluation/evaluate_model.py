import torch
import torch.nn as nn
import os
import sys
import joblib
import numpy as np
import librosa
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import glob

# Add mers to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'mers'))

# Constants (Must match training!)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Audio Constants
SAMPLE_RATE = 16000
DURATION = 3.0 
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
N_MFCC = 128
HOP_LENGTH = 512

def evaluate_visual(data_dir):
    print("\n[Visual Model Evaluation]")
    print(f"Dataset: {data_dir}")
    
    # Use absolute path based on PROJECT_ROOT defined in main block or passed in
    # but since this function is called from main block, let's just use the known relative path from root
    # assuming script is run from project root.
    
    # Better: Use the define PROJECT_ROOT if available or re-calculate
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "assets", "models", "visual_efficientnet.pth")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    # Transforms
    val_transforms = transforms.Compose([
        transforms.Resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # Load Data
    # Check for split
    if os.path.exists(os.path.join(data_dir, 'val')):
        data_dir = os.path.join(data_dir, 'val')
        print("Using 'val' subdirectory.")
    
    try:
        dataset = datasets.ImageFolder(data_dir, transform=val_transforms)
    except Exception as e:
        print(f"Error loading visual dataset: {e}")
        return
        
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Classes: {dataset.classes}")
    
    # Load Model
    print("Loading Model...")
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    # Assuming 7 classes as per settings.py
    model.classifier[1] = nn.Linear(in_features, 7)  
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading dict: {e}")
        return

    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    print("Running Inference...")
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"\nVisual Accuracy: {acc*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=dataset.classes))


class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0) 
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

def evaluate_audio(data_dir):
    print("\n[Audio Model Evaluation]")
    print(f"Dataset root: {data_dir}")
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "assets", "models", "audio_resnet18.pth")
    encoder_path = os.path.join(project_root, "assets", "models", "label_encoder_cnn.pkl")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
        
    # Preprocess Data (Similar to train_audio_resnet.py)
    # We will pick 20 samples per class to save time, or all if feasible.
    # Let's try to load all validate set if possible, or just raw folders.
    
    X = []
    y_labels = []
    
    # Expected directories: angry, happy, etc.
    if os.path.exists(os.path.join(data_dir, 'telugu')):
         data_dir = os.path.join(data_dir, 'telugu') # Adjust if nested
         
    classes = os.listdir(data_dir)
    print(f"Found classes folders: {classes}")
    
    print("Preprocessing Audio (this may take time)...")
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path): continue
        
        files = glob.glob(os.path.join(cls_path, "*"))
        # Limit for speed if needed, but user wants accuracy so take all?
        # Let's take all.
        
        for file_path in tqdm(files, desc=cls):
            if not file_path.lower().endswith(('.wav', '.mp3')): continue
            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                if len(signal) > SAMPLES_PER_TRACK:
                    signal = signal[:SAMPLES_PER_TRACK]
                else:
                    pad = SAMPLES_PER_TRACK - len(signal)
                    signal = np.pad(signal, (0, pad), mode='constant')
                    
                melspec = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=N_MFCC, hop_length=HOP_LENGTH)
                melspec = librosa.power_to_db(melspec, ref=np.max)
                
                mean = np.mean(melspec); std = np.std(melspec)
                if std > 0: melspec = (melspec - mean)/std
                else: melspec = (melspec - mean)
                
                X.append(melspec)
                y_labels.append(cls) # Raw label
            except: pass

    if not X:
        print("No audio data found.")
        return

    X = np.array(X)
    
    # Encode Labels
    le = joblib.load(encoder_path)
    # Map raw directory names to encoder classes if needed
    # dir: 'angry' -> encoder: 'Angry' usually
    # Try case insensitive mapping
    y_encoded = []
    for label in y_labels:
        # Simple mapping heuristic
        found = False
        for cls_name in le.classes_:
            if label.lower() == cls_name.lower():
                y_encoded.append(le.transform([cls_name])[0])
                found = True
                break
        if not found:
             # Default or skip?
             # Let's assume neutral
             # Or skip
             pass
    
    # If lengths differ due to mapping failure, we have an issue.
    # For now assume mostly match.
    if len(y_encoded) != len(X):
        # Fallback exact match or adjust X
        # Re-loop to be safe? 
        # Actually simpler: just iterate and append both if valid
        pass 

    # Re-do encoding safely
    X_final = []
    y_final = []
    
    for i, label in enumerate(y_labels):
        for cls_name in le.classes_:
            if label.lower() == cls_name.lower():
                X_final.append(X[i])
                y_final.append(le.transform([cls_name])[0])
                break
    
    if not X_final:
        print("Could not map any audio labels to model classes.")
        return
        
    X = np.array(X_final)
    y = np.array(y_final)

    dataset = AudioDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load Model
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(le.classes_))
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    print("Running Audio Inference...")
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAudio Accuracy: {acc*100:.2f}%")
    
    # Debug: Check unique labels
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    print(f"Unique encoded labels in GT: {unique_true}")
    print(f"Unique encoded labels in Pred: {unique_pred}")
    print(f"Encoder classes: {le.classes_}")
    
    # Ensure target_names matches the indices in y_true/y_pred
    # The encoder classes should map 0..N-1
    # If y_true contains indices not in 0..N-1, that's an issue.
    # classification_report expects target_names to correspond to sorted unique labels if labels param is not provided,
    # or exactly the labels param.
    
    try:
        print(classification_report(y_true, y_pred, target_names=le.classes_))
    except Exception as e:
        print(f"Error in classification_report: {e}")
        print("Attempting automatic report generation...")
        print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    # Path configuration
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Visual
    visual_data = os.path.join(PROJECT_ROOT, "datasets", "Images")
    if os.path.exists(visual_data):
        evaluate_visual(visual_data)
    else:
        print("Visual dataset not found at datasets/Images")

    # 2. Audio
    audio_data = os.path.join(PROJECT_ROOT, "datasets", "telugu_audio")
    if os.path.exists(audio_data):
        evaluate_audio(audio_data)
    else:
        print("Audio dataset not found at datasets/telugu_audio")
