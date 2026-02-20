
import os
import sys
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import joblib

# Define project_dir at the top
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add backend paths
sys.path.append(os.path.join(project_dir, "backend"))
sys.path.append(os.path.join(project_dir, "backend", "src"))
from config.settings import MODELS_DIR, EMOTIONS

# Configuration
ORIGINAL_DATA_DIR = os.path.join(project_dir, 'datasets', 'telugu_audio', 'telugu')
SYNTHETIC_DATA_DIR = os.path.join(project_dir, 'datasets', 'telugu_audio_synthetic', 'telugu')
MODELS_DIR = os.path.join(project_dir, "backend", "assets", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Audio Config
SAMPLE_RATE = 16000
DURATION = 3.0 # Increased duration context
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
N_MFCC = 128 # Using Mels instead of MFCC for ResNet (Frequency resolution)
HOP_LENGTH = 512 

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # ResNet expects (3, H, W) usually, but we will modify input layer to (1, H, W)
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0) 
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

def load_and_preprocess_data():
    X = []
    y = []
    
    dir_to_label = {
        'angry': 'Angry',
        'happy': 'Happy',
        'nuetral': 'Neutral',
        'sad': 'Sad',
        'suprised': 'Surprised'
    }
    
    directories = [ORIGINAL_DATA_DIR, SYNTHETIC_DATA_DIR]
    
    print("Loading and preprocessing data (Mel-Spectrograms)...")
    for data_dir in directories:
        if not os.path.exists(data_dir):
            continue
            
        for dir_name, label in dir_to_label.items():
            emotion_path = os.path.join(data_dir, dir_name)
            if not os.path.exists(emotion_path):
                continue
                
            files = glob.glob(os.path.join(emotion_path, "*"))
            
            for file_path in tqdm(files, desc=f"{label}"):
                if not file_path.lower().endswith(('.wav', '.mp3')):
                    continue
                    
                try:
                    # Load audio
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    
                    # Fix length to SAMPLES_PER_TRACK
                    if len(signal) > SAMPLES_PER_TRACK:
                        signal = signal[:SAMPLES_PER_TRACK]
                    else:
                        pad_width = SAMPLES_PER_TRACK - len(signal)
                        signal = np.pad(signal, (0, pad_width), mode='constant')
                    
                    # Compute Log-Mel Spectrogram (Rich features for ResNet)
                    melspec = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=N_MFCC, hop_length=HOP_LENGTH)
                    melspec = librosa.power_to_db(melspec, ref=np.max)
                    
                    # Normalization (Crucial for Neural Nets)
                    mean = np.mean(melspec)
                    std = np.std(melspec)
                    if std > 0:
                        melspec = (melspec - mean) / std
                    else:
                        melspec = (melspec - mean)
                    
                    # Pad width if needed (ResNet likes square-ish)
                    # Current width ~ 93 for 3s. Let's pad to fixed size if needed or leave flexible (AdaptivePooling handles it)
                    
                    X.append(melspec)
                    y.append(label)
                    
                except Exception as e:
                    # print(f"Error: {e}")
                    pass
                    
    return np.array(X), np.array(y)

def train_resnet():
    # 1. Load Data
    X_raw, y_raw = load_and_preprocess_data()
    
    if len(X_raw) == 0:
        print("No data found!")
        return

    # 2. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder_cnn.pkl"))
    
    print(f"Classes: {le.classes_}")
    num_classes = len(le.classes_)
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
    
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Model Setup (ResNet18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = models.resnet18(pretrained=True)
    
    # Modify first layer for 1 channel (Grayscale/Spectrogram)
    # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify FC layer for num_classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # 5. Training Loop
    epochs = 40
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # Step Scheduler
        scheduler.step(val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "audio_resnet18.pth"))
            
    print(f"Training Complete. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_resnet()
