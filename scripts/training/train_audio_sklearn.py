import os
import sys
import numpy as np
import librosa
import joblib
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Define project_dir at the top
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add src to path
sys.path.append(os.path.join(project_dir, "backend", "src"))

from config.settings import EMOTIONS, SAMPLE_RATE, AUDIO_CHUNK_DURATION
from mers.core.features import extract_audio_features

# Re-use the dataset loader logic but simplified for sklearn
# We don't need a Torch Dataset class, just a function to load X, y

def load_data(data_root, emotions):
    X = []
    y = []
    
    class_to_idx = {e: i for i, e in enumerate(emotions)}
    print(f"Emotion Map: {class_to_idx}")
    
    # We will reuse the file finding logic from previous script via a helper or just copy-paste for safety/speed
    # Let's quickly traverse the known directories since we know the structure now
    
    files = []
    labels = []
    
    # 1. Crema
    path = os.path.join(data_root, "Crema")
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith('.wav'):
                # 1001_DFA_ANG_XX.wav
                parts = f.split('_')
                if len(parts) >= 3:
                    code = parts[2]
                    emo_map = {"ANG": "Angry", "DIS": "Disgust", "FEA": "Fear", "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"}
                    if code in emo_map and emo_map[code] in class_to_idx:
                        files.append(os.path.join(path, f))
                        labels.append(class_to_idx[emo_map[code]])
                        
    # 2. Ravdess
    path = os.path.join(data_root, "Ravdess")
    if os.path.exists(path):
        for root, _, filenames in os.walk(path):
            for f in filenames:
                if f.endswith('.wav'):
                    # 03-01-01-01-01-01-01.wav
                    parts = f.split('-')
                    if len(parts) >= 3:
                        code = parts[2]
                        emo_map = {"01": "Neutral", "02": "Neutral", "03": "Happy", "04": "Sad", "05": "Angry", "06": "Fear", "07": "Disgust", "08": "Surprise"}
                        if code in emo_map and emo_map[code] in class_to_idx:
                            files.append(os.path.join(root, f))
                            labels.append(class_to_idx[emo_map[code]])

    # 3. Savee
    path = os.path.join(data_root, "Savee")
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith('.wav'):
                # DC_a01.wav
                parts = f.split('_')
                if len(parts) >= 2:
                    rest = parts[1]
                    code = "".join([c for c in rest if c.isalpha()])
                    emo_map = {"a": "Angry", "d": "Disgust", "f": "Fear", "h": "Happy", "n": "Neutral", "sa": "Sad", "su": "Surprise"}
                    if code in emo_map and emo_map[code] in class_to_idx:
                        files.append(os.path.join(path, f))
                        labels.append(class_to_idx[emo_map[code]])

    # 4. Tess
    path = os.path.join(data_root, "Tess")
    if os.path.exists(path):
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                emo = None
                lower = folder.lower()
                if "angry" in lower: emo = "Angry"
                elif "disgust" in lower: emo = "Disgust"
                elif "fear" in lower: emo = "Fear"
                elif "happy" in lower: emo = "Happy"
                elif "neutral" in lower: emo = "Neutral"
                elif "sad" in lower: emo = "Sad"
                elif "surprise" in lower: emo = "Surprise"
                
                if emo and emo in class_to_idx:
                    for f in os.listdir(folder_path):
                        if f.endswith('.wav'):
                            files.append(os.path.join(folder_path, f))
                            labels.append(class_to_idx[emo])

    print(f"Found {len(files)} files.")
    
    # Extract Features
    print("Extracting features (Prosody + MFCC)... This may take a while.")
    for f, label in tqdm(zip(files, labels), total=len(files)):
        try:
            # Load with librosa
            audio_signal, sr = librosa.load(f, sr=SAMPLE_RATE, duration=AUDIO_CHUNK_DURATION)
            
            # Extract
            feat_vec, _ = extract_audio_features(audio_signal, sr)
            X.append(feat_vec)
            y.append(label)
        except Exception as e:
            # print(f"Error {f}: {e}")
            pass
            
    return np.array(X), np.array(y)

def train_sklearn_model():
    data_root = os.path.join(project_dir, "datasets")
    
    print(f"Data Root: {data_root}")
    X, y = load_data(data_root, EMOTIONS)
    
    if len(X) == 0:
        print("No data found!")
        return

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    models_dir = os.path.join(project_dir, "backend", "assets", "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "audio_scaler.pkl"))
    print("Scaler saved.")

    # 4. Train Model (MLP as requested)
    print("Training MLP Classifier...")
    # Lightweight MLP: 2 hidden layers, 64/32 units
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, alpha=1e-4,
                        solver='adam', verbose=10, random_state=1,
                        learning_rate_init=1e-3)
                        
    clf.fit(X_train_scaled, y_train)
    
    # 5. Evaluate
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=EMOTIONS))
    
    # 6. Save
    model_path = os.path.join(models_dir, "audio_mlp.pkl")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Define SCALER_PATH here if not in config
    models_dir = os.path.join(project_dir, "backend", "assets", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    SCALER_PATH = os.path.join(models_dir, "audio_scaler.pkl")
    train_sklearn_model()
