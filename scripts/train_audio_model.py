
import os
import sys
import numpy as np
import librosa
import joblib
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add src to path to import features
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, "backend", "src"))
from mers.core.features import extract_audio_features
from config.settings import EMOTIONS, MODELS_DIR

# Configuration
ORIGINAL_DATA_DIR = os.path.join(project_dir, 'datasets', 'telugu_audio', 'telugu')
SYNTHETIC_DATA_DIR = os.path.join(project_dir, 'datasets', 'telugu_audio_synthetic', 'telugu')

# Ensure models directory exists
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def load_data():
    X = []
    y = []
    
    # We use the emotions list from config, but we need to match directory names
    # Config EMOTIONS: ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised']
    # Directory Names: ['angry', 'happy', 'nuetral', 'sad', 'suprised']
    
    # Map directory names to Config Emotion Labels
    dir_to_label = {
        'angry': 'Angry',
        'happy': 'Happy',
        'nuetral': 'Neutral',
        'sad': 'Sad',
        'suprised': 'Surprised'
    }
    
    directories = [ORIGINAL_DATA_DIR, SYNTHETIC_DATA_DIR]
    
    print("Loading data...")
    for data_dir in directories:
        if not os.path.exists(data_dir):
            print(f"Warning: Directory not found: {data_dir}")
            continue
            
        for dir_name, label in dir_to_label.items():
            emotion_path = os.path.join(data_dir, dir_name)
            if not os.path.exists(emotion_path):
                continue
                
            files = glob.glob(os.path.join(emotion_path, "*"))
            print(f"Loading {len(files)} files for {label} from {os.path.basename(data_dir)}")
            
            for file_path in tqdm(files, desc=f"Processing {label}"):
                if not file_path.lower().endswith(('.wav', '.mp3')):
                    continue
                    
                try:
                    # Load audio
                    data, sr = librosa.load(file_path, sr=None)
                    
                    # Extract features
                    # extract_audio_features returns (vector, features_dict)
                    # We utilize the vector directly which is flattened and ready for ML
                    feat_vec, _ = extract_audio_features(data, sr=sr)
                    
                    # Handle NaNs
                    if np.isnan(feat_vec).any():
                        feat_vec = np.nan_to_num(feat_vec)
                        
                    X.append(feat_vec)
                    y.append(label)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y)

def train_model():
    X, y = load_data()
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {le.classes_}")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Model
    print("Training MLP Classifier...")
    # MLP Architecture:
    # Input -> [256] -> [128] -> [64] -> Output
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True
    )
    
    clf.fit(X_train_scaled, y_train)
    
    # Evaluation
    print("\nEvaluating model...")
    y_pred = clf.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save Artifacts
    print(f"Saving model and scaler to {MODELS_DIR}...")
    joblib.dump(clf, os.path.join(MODELS_DIR, "audio_mlp.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "audio_scaler.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl")) # Save LE to know class mapping later
    
    print("Training completed successfully.")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
