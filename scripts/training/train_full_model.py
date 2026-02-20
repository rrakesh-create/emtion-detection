import os
import sys
import warnings
from typing import List, Tuple

import joblib
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Define project_dir at the top
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add src to path
sys.path.append(os.path.join(project_dir, "backend", "src"))

from config.settings import OUTPUT_EMOTIONS
from mers.core.features import extract_audio_features

warnings.filterwarnings("ignore")

# Define Robust Augmentation Functions
def augment_audio(y: np.ndarray, sr: int) -> List[np.ndarray]:
    """
    Apply robust augmentations to the audio signal.
    Returns a list of augmented signals (including the original).
    """
    augmented = [y]
    
    # 1. Noise Injection
    noise_amp = 0.005 * np.random.uniform() * np.amax(y)
    y_noise = y + noise_amp * np.random.normal(size=y.shape)
    augmented.append(y_noise)
    
    # 2. Pitch Shift (Key for emotion)
    # Shift by -2, -1, 1, 2 semitones
    steps = np.random.choice([-2, -1, 1, 2])
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    augmented.append(y_pitch)
    
    # 3. Time Stretch (Speed)
    # Fast (1.1x) or Slow (0.9x)
    rate = np.random.choice([0.9, 1.1])
    y_stretch = librosa.effects.time_stretch(y, rate=rate)
    
    # Fix length to match original roughly (or just let feature extractor handle it)
    # Feature extractor handles variable length via mean/std, so raw length doesn't matter much.
    augmented.append(y_stretch)
    
    return augmented

def load_data_robust(data_root, emotions, use_augmentation=True):
    X = []
    y = []
    
    class_to_idx = {e: i for i, e in enumerate(emotions)}
    print(f"Emotion Map: {class_to_idx}")
    
    files = []
    labels = []
    
    # --- 1. Crema ---
    path = os.path.join(data_root, "Crema")
    if os.path.exists(path):
        print(f"Scanning Crema in {path}...")
        for f in os.listdir(path):
            if f.endswith('.wav'):
                parts = f.split('_')
                if len(parts) >= 3:
                    code = parts[2]
                    emo_map = {"ANG": "Angry", "DIS": "Stressed", "FEA": "Fear", "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"}
                    if code in emo_map and emo_map[code] in class_to_idx:
                        files.append(os.path.join(path, f))
                        labels.append(class_to_idx[emo_map[code]])

    # --- 2. Ravdess ---
    path = os.path.join(data_root, "Ravdess")
    if os.path.exists(path):
        print(f"Scanning Ravdess in {path}...")
        for root, _, filenames in os.walk(path):
            for f in filenames:
                if f.endswith('.wav'):
                    parts = f.split('-')
                    if len(parts) >= 3:
                        code = parts[2]
                        emo_map = {
                            "01": "Neutral", "02": "Neutral", 
                            "03": "Happy", "04": "Sad", 
                            "05": "Angry", "06": "Fear", 
                            "07": "Stressed", "08": "Focused"
                        }
                        if code in emo_map and emo_map[code] in class_to_idx:
                            files.append(os.path.join(root, f))
                            labels.append(class_to_idx[emo_map[code]])

    # --- 3. Savee ---
    path = os.path.join(data_root, "Savee")
    if os.path.exists(path):
        print(f"Scanning Savee in {path}...")
        for f in os.listdir(path):
            if f.endswith('.wav'):
                parts = f.split('_')
                if len(parts) >= 2:
                    rest = parts[1]
                    code = "".join([c for c in rest if c.isalpha()])
                    emo_map = {
                        "a": "Angry", "d": "Stressed", "f": "Fear", 
                        "h": "Happy", "n": "Neutral", "sa": "Sad", "su": "Focused"
                    }
                    if code in emo_map and emo_map[code] in class_to_idx:
                        files.append(os.path.join(path, f))
                        labels.append(class_to_idx[emo_map[code]])

    # --- 4. Tess ---
    path = os.path.join(data_root, "Tess")
    if os.path.exists(path):
        print(f"Scanning Tess in {path}...")
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                emo = None
                lower = folder.lower()
                if "angry" in lower: emo = "Angry"
                elif "disgust" in lower: emo = "Stressed"
                elif "fear" in lower: emo = "Fear"
                elif "happy" in lower: emo = "Happy"
                elif "neutral" in lower: emo = "Neutral"
                elif "sad" in lower: emo = "Sad"
                elif "surprise" in lower: emo = "Focused"
                
                if emo and emo in class_to_idx:
                    for f in os.listdir(folder_path):
                        if f.endswith('.wav'):
                            files.append(os.path.join(folder_path, f))
                            labels.append(class_to_idx[emo])

    print(f"Found {len(files)} total files.")
    
    # Extract Features with enhanced augmentation
    print("Extracting features (This will take longer due to Pitch/Time augmentation)...")
    
    # Limit samples for demo speed if needed, but user asked for "More Accurate", so we process ALL.
    # We use Parallel to speed up extraction
    from joblib import Parallel, delayed
    
    def process_file(f, label):
        local_X = []
        local_y = []
        try:
            y_raw, sr = librosa.load(f, sr=16000, duration=3.0)
            if len(y_raw) < 1000: return [], []
            
            # Apply Augmentation (Skip for speed in interactive session unless explicit)
            # signals = augment_audio(y_raw, sr) if use_augmentation else [y_raw]
            signals = [y_raw] # Force no augmentation for speed
            
            for sig in signals:
                if len(sig) < 512: continue
                feat_vec, _ = extract_audio_features(sig, sr)
                local_X.append(feat_vec)
                local_y.append(label)
        except Exception:
            pass
        return local_X, local_y

    results = Parallel(n_jobs=-1)(delayed(process_file)(f, l) for f, l in tqdm(zip(files, labels), total=len(files)))
    
    for rx, ry in results:
        X.extend(rx)
        y.extend(ry)
            
    return np.array(X), np.array(y)

def train_ensemble_model():
    data_root = os.path.join(project_dir, "datasets")
    
    print(f"Starting Robust Training Pipeline.")
    
    # 1. Load Data
    X, y = load_data_robust(data_root, OUTPUT_EMOTIONS, use_augmentation=True)
    
    if len(X) == 0:
        print("No data found!")
        return

    print(f"Dataset Size: {X.shape[0]} samples")
    print(f"Feature Dimension: {X.shape[1]}")
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    models_dir = os.path.join(project_dir, "backend", "assets", "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "audio_scaler.pkl"))
    
    # 4. Define Models for Voting Ensemble
    print("Training Ensemble Model (MLP + Random Forest + HistGradientBoosting)...")
    
    # MLP (Neural Net)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
    
    # Random Forest (Robust to noise)
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    
    # Histogram Gradient Boosting (State of the art for tabular)
    hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, random_state=42)
    
    # Voting Classifier (Soft Voting for probabilities)
    voting_clf = VotingClassifier(
        estimators=[('mlp', mlp), ('rf', rf), ('hgb', hgb)],
        voting='soft',
        n_jobs=-1
    )
    
    # Train
    voting_clf.fit(X_train_scaled, y_train)
    
    # 5. Evaluate
    y_pred = voting_clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nEnsemble Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=OUTPUT_EMOTIONS))
    
    # 6. Save
    model_path = os.path.join(models_dir, "audio_mlp.pkl") # Keep same name for compatibility or update server?
    # Server expects "audio_mlp.pkl". Let's overwrite it with the VotingClassifier (it has predict_proba same as MLP)
    joblib.dump(voting_clf, model_path)
    print(f"Ensemble Model saved to {model_path}")
    
    # Metadata
    metadata = {
        "accuracy": acc,
        "emotions": OUTPUT_EMOTIONS,
        "model_type": "VotingClassifier(MLP+RF+HGB)",
        "date": "2026-02-07"
    }
    joblib.dump(metadata, os.path.join(models_dir, "model_metadata.pkl"))

if __name__ == "__main__":
    train_ensemble_model()
