import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import sys
import os
import glob
from tqdm import tqdm

# Define project_dir at the top
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add src to path
sys.path.append(os.path.join(project_dir, "backend", "src"))

from config.settings import EMOTIONS, SAMPLE_RATE, AUDIO_CHUNK_DURATION, N_MFCC, NUM_CLASSES, DEVICE, AUDIO_MODEL_PATH, MFCC_MEAN_PATH, MFCC_STD_PATH
from mers.core.audio_model import AudioEmotionModel

class UnifiedAudioDataset(Dataset):
    def __init__(self, data_root, emotions, sr=SAMPLE_RATE, duration=AUDIO_CHUNK_DURATION, augment=False):
        self.files = []
        self.labels = []
        self.sr = sr
        self.duration = duration
        self.target_len = int(sr * duration)
        self.class_to_idx = {e: i for i, e in enumerate(emotions)}
        
        print(f"Scanning datasets in {data_root}...")
        self._load_crema(os.path.join(data_root, "Crema"))
        self._load_ravdess(os.path.join(data_root, "Ravdess"))
        self._load_savee(os.path.join(data_root, "Savee"))
        self._load_tess(os.path.join(data_root, "Tess"))
        
        original_count = len(self.files)
        self.is_augmented = [False] * original_count
        
        if augment:
            print(f"Augmenting dataset (Doubling size from {original_count} to {original_count * 2})...")
            self.files.extend(self.files) # Duplicate files
            self.labels.extend(self.labels) # Duplicate labels
            self.is_augmented.extend([True] * original_count) # Mark duplicates for augmentation
            
        print(f"Total samples: {len(self.files)}")

    def _augment_audio(self, y):
        """Apply random augmentation: Noise, Stretch, or Pitch Shift"""
        choice = np.random.randint(0, 3)
        
        try:
            if choice == 0: # Noise
                noise_amp = 0.005 * np.random.uniform() * np.amax(y)
                y = y + noise_amp * np.random.normal(size=y.shape[0])
            elif choice == 1: # Stretch
                rate = np.random.uniform(0.8, 1.2)
                y = librosa.effects.time_stretch(y, rate=rate)
            elif choice == 2: # Pitch Shift
                steps = np.random.randint(-2, 3)
                y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=steps)
        except Exception as e:
            # Fallback if augmentation fails (e.g. too short)
            pass
            
        return y

    def _load_crema(self, path):
        if not os.path.exists(path):
            print(f"Crema path not found: {path}")
            return
            
        print(f"Scanning Crema in {path}...")
        # Crema filenames: 1001_DFA_ANG_XX.wav
        # Map: ANG->Angry, DIS->Disgust, FEA->Fear, HAP->Happy, NEU->Neutral, SAD->Sad
        count = 0 
        emotion_map = {
            "ANG": "Angry", "DIS": "Disgust", "FEA": "Fear", 
            "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"
        }
        
        for f in glob.glob(os.path.join(path, "*.wav")):
            filename = os.path.basename(f)
            parts = filename.split('_')
            if len(parts) >= 3:
                code = parts[2]
                if code in emotion_map:
                    emo = emotion_map[code]
                    if emo in self.class_to_idx:
                        self.files.append(f)
                        self.labels.append(self.class_to_idx[emo])
                        count += 1
        print(f"Loaded {count} files from Crema")

    def _load_ravdess(self, path):
        if not os.path.exists(path):
            print(f"Ravdess path not found: {path}")
            return
            
        print(f"Scanning Ravdess in {path}...")
        # Ravdess filenames: 03-01-01-01-01-01-01.wav
        # 3rd part is emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=surprise
        count = 0
        emotion_map = {
            "01": "Neutral", "02": "Neutral", "03": "Happy", 
            "04": "Sad", "05": "Angry", "06": "Fear", 
            "07": "Disgust", "08": "Surprise"
        }
        
        # Search recursively for Actor_* folders
        # glob recursive might be tricky on windows sometimes depending on python version
        # Try specific pattern
        for f in glob.glob(os.path.join(path, "**", "*.wav"), recursive=True):
            filename = os.path.basename(f)
            parts = filename.split('-')
            if len(parts) >= 3:
                code = parts[2]
                if code in emotion_map:
                    emo = emotion_map[code]
                    if emo in self.class_to_idx:
                        self.files.append(f)
                        self.labels.append(self.class_to_idx[emo])
                        count += 1
        print(f"Loaded {count} files from Ravdess")

    def _load_savee(self, path):
        if not os.path.exists(path):
            print(f"Savee path not found: {path}")
            return
            
        print(f"Scanning Savee in {path}...")
        # Savee filenames: DC_a01.wav
        # a=anger, d=disgust, f=fear, h=happiness, n=neutral, sa=sadness, su=surprise
        count = 0
        emotion_map = {
            "a": "Angry", "d": "Disgust", "f": "Fear", 
            "h": "Happy", "n": "Neutral", "sa": "Sad", "su": "Surprise"
        }
        
        for f in glob.glob(os.path.join(path, "*.wav")):
            filename = os.path.basename(f)
            # Code is usually between _ and digits, e.g. DC_a01.wav -> 'a'
            # Or JK_sa01.wav -> 'sa'
            parts = filename.split('_')
            if len(parts) >= 2:
                # Part 2 is like "a01.wav" or "sa01.wav"
                rest = parts[1]
                # Remove extension
                rest = os.path.splitext(rest)[0]
                # Extract letters
                code = "".join([c for c in rest if c.isalpha()])
                
                if code in emotion_map:
                    emo = emotion_map[code]
                    if emo in self.class_to_idx:
                        self.files.append(f)
                        self.labels.append(self.class_to_idx[emo])
                        count += 1
                else: 
                     # Debug skipped files
                     # print(f"Skipped Savee: {filename} code={code}")
                     pass

        print(f"Loaded {count} files from Savee")

    def _load_tess(self, path):
        if not os.path.exists(path):
            print(f"Tess path not found: {path}")
            return
            
        print(f"Scanning Tess in {path}...")
        # Tess: Folders like OAF_Fear, YAF_angry
        count = 0
        
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                # Parse folder name
                # usually OAF_Emotion or YAF_emotion
                parts = folder.split('_')
                if len(parts) >= 2:
                    emotion_str = parts[-1].lower()
                    if emotion_str == "pleasant": # unexpected case fix
                        continue 
                    if "pleasant" in emotion_str and "surprise" in emotion_str:
                         emotion_str = "surprise"
                    
                    # Manual mapping
                    emo = None
                    if "angry" in emotion_str: emo = "Angry"
                    elif "disgust" in emotion_str: emo = "Disgust"
                    elif "fear" in emotion_str: emo = "Fear"
                    elif "happy" in emotion_str: emo = "Happy"
                    elif "neutral" in emotion_str: emo = "Neutral"
                    elif "sad" in emotion_str: emo = "Sad"
                    elif "surprise" in emotion_str: emo = "Surprise"
                    
                    if emo and emo in self.class_to_idx:
                         for f in glob.glob(os.path.join(folder_path, "*.wav")):
                            self.files.append(f)
                            self.labels.append(self.class_to_idx[emo])
                            count += 1
        print(f"Loaded {count} files from Tess")
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels[idx]
        
        try:
            # Add retries or safer loading
            y, sr = librosa.load(file, sr=self.sr, duration=self.duration)
            
            # Apply Augmentation if needed
            if self.is_augmented[idx]:
                y = self._augment_audio(y)
            
            # Pad or Truncate
            if len(y) < self.target_len:
                y = np.pad(y, (0, self.target_len - len(y)))
            else:
                y = y[:self.target_len]
                
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # Fallback for bad files
            print(f"Error loading {file}: {e}")
            return torch.zeros((1, N_MFCC, int(self.duration*self.sr/512)+1)), torch.tensor(label, dtype=torch.long)

def train_audio_model(data_path, epochs=30, batch_size=32):
    print(f"Training Audio Model on {DEVICE}")
    
    # Enable augmentation for training
    dataset = UnifiedAudioDataset(data_path, EMOTIONS, augment=True)
    if len(dataset) == 0:
        print("No audio files found! Check dataset path.")
        return

    # Split train/val
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # Set to 0 for Windows stability
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = AudioEmotionModel(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Calculate global stats (approximate with subset)
    if not os.path.exists(MFCC_MEAN_PATH):
        print("Calculating dataset stats for normalization...")
        all_mfccs = []
        # Use first 3000 samples for stats to be more representative
        indices = np.random.choice(len(dataset), min(len(dataset), 3000), replace=False)
        for i in tqdm(indices):
            mfcc, _ = dataset[i]
            all_mfccs.append(mfcc.numpy())
        
        all_mfccs = np.concatenate(all_mfccs, axis=0)
        mean = np.mean(all_mfccs, axis=(0, 1, 3))
        std = np.std(all_mfccs, axis=(0, 1, 3))
        
        np.save(MFCC_MEAN_PATH, mean)
        np.save(MFCC_STD_PATH, std)
        print(f"Saved MFCC stats to {MFCC_MEAN_PATH}")
    else:
        print("Loading existing MFCC stats...")
        mean = np.load(MFCC_MEAN_PATH)
        std = np.load(MFCC_STD_PATH)
    
    mean_torch = torch.tensor(mean, dtype=torch.float32).view(1, 1, 40, 1).to(DEVICE)
    std_torch = torch.tensor(std, dtype=torch.float32).view(1, 1, 40, 1).to(DEVICE)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Normalize
            inputs = (inputs - mean_torch) / (std_torch + 1e-6)
            
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
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                inputs = (inputs - mean_torch) / (std_torch + 1e-6)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}: Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), AUDIO_MODEL_PATH)
            print(f"New best model saved! ({val_acc:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default path
        default_path = os.path.join(project_dir, "datasets")
        if os.path.exists(default_path):
            train_audio_model(default_path)
        else:
            print("Usage: python train_audio.py <path_to_datasets_root>")
    else:
        train_audio_model(sys.argv[1])
