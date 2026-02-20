import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
from torchvision import models
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '..', 'telugu_audio', 'telugu')
MODELS_DIR = os.path.join(BASE_DIR, "assets", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "audio_resnet18.pth")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder_cnn.pkl")

# Config
SAMPLE_RATE = 16000
DURATION = 3.0
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
N_MFCC = 128
HOP_LENGTH = 512

def load_model(encoder_path, model_path, device):
    le = joblib.load(encoder_path)
    num_classes = len(le.classes_)
    
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, le

def preprocess_audio(file_path):
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        if len(signal) > SAMPLES_PER_TRACK:
            signal = signal[:SAMPLES_PER_TRACK]
        else:
            pad_width = SAMPLES_PER_TRACK - len(signal)
            signal = np.pad(signal, (0, pad_width), mode='constant')
        
        melspec = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=N_MFCC, hop_length=HOP_LENGTH)
        melspec = librosa.power_to_db(melspec, ref=np.max)
        
        mean = np.mean(melspec)
        std = np.std(melspec)
        if std > 0:
            melspec = (melspec - mean) / std
        else:
            melspec = (melspec - mean)
            
        return torch.tensor(melspec, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    except Exception as e:
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("Model or Encoder not found!")
        return

    print("Loading Model...")
    model, le = load_model(ENCODER_PATH, MODEL_PATH, device)
    print(f"Classes: {le.classes_}")
    
    print("Evaluating on Telugu Dataset...")
    
    y_true = []
    y_pred = []
    
    dir_to_label = {
        'angry': 'Angry',
        'happy': 'Happy',
        'nuetral': 'Neutral',
        'sad': 'Sad',
        'suprised': 'Surprised'
    }
    
    total_files: int = 0
    
    # Cap files per emotion to speed up check (e.g., 20 per class)
    MAX_PER_CLASS = 20

    for dir_name, label in dir_to_label.items():
        emotion_path = os.path.join(DATA_DIR, dir_name)
        if not os.path.exists(emotion_path):
            continue
            
        files = glob.glob(os.path.join(emotion_path, "*"))
        # Random shuffle to get variety
        np.random.shuffle(files)
        files = files[:MAX_PER_CLASS]
        
        for file_path in files:
            if not file_path.lower().endswith(('.wav', '.mp3')):
                continue
                
            input_tensor = preprocess_audio(file_path)
            if input_tensor is not None:
                input_tensor = input_tensor.to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted_idx = torch.max(outputs, 1)
                    predicted_label = le.classes_[predicted_idx.item()]
                    
                y_true.append(label)
                y_pred.append(predicted_label)
                total_files += 1

    if total_files == 0:
        print("No files found!")
        return

    acc = accuracy_score(y_true, y_pred)
    print(f"\nChecked {total_files} random samples from dataset.")
    print(f"Accuracy: {acc*100:.2f}%")
    
    print("\nSample Predictions:")
    for i in range(min(5, len(y_true))):
        print(f"True: {y_true[i]} | Pred: {y_pred[i]}")

if __name__ == "__main__":
    main()
