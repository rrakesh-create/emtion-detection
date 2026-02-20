import os
import json

# --- System Configuration ---
APP_NAME = "MERS"
VERSION = "1.3.0"

# --- Hardware / Device ---
DEVICE = "cpu" # Default to CPU for lightweight models

# --- Internal Model Classes (Do Not Change - tied to trained models) ---
# RAVDESS/FER usually have 7 classes
MODEL_EMOTIONS = ["Angry", "Happy", "Sad", "Neutral", "Fear", "Surprise", "Disgust"]
NUM_CLASSES = len(MODEL_EMOTIONS)

# --- User Output Classes (Requested) ---
OUTPUT_EMOTIONS = ["Happy", "Sad", "Angry", "Fear", "Neutral", "Surprise", "Disgust"]

# Global Emotions List (Unified)
EMOTIONS = OUTPUT_EMOTIONS

# --- Mapping: Model -> Output ---
# Heuristic mapping to satisfy user requirements without retraining immediately
EMOTION_MAPPING = {
    "Angry": "Angry",
    "Happy": "Happy",
    "Sad": "Sad",
    "Neutral": "Neutral",
    "Fear": "Fear",
    "Surprise": "Surprise", 
    "Disgust": "Disgust"  
}

# --- Paths ---
# Resolve Project Root from settings.py location: backend/config/settings.py -> backend/config -> backend -> root
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CONFIG_DIR) # backend
PROJECT_ROOT = os.path.dirname(BASE_DIR) # project root
MODELS_DIR = os.path.join(PROJECT_ROOT, "assets", "models")
SRC_DIR = os.path.join(BASE_DIR, "src")
LOG_FILE = os.path.join(BASE_DIR, "assets", "logs", "emotion_log.txt")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

# Ensure logs directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Model Paths (Scikit-learn)
AUDIO_MODEL_PATH = os.path.join(MODELS_DIR, "audio_mlp.pkl")
AUDIO_SCALER_PATH = os.path.join(MODELS_DIR, "audio_scaler.pkl")

# --- Load External Config ---
CONFIG = {
    "WEBCAM_ID": 0,
    "FACE_DETECTION_THRESHOLD": 0.95,
    "VISUAL_INFERENCE_INTERVAL": 0.3,
    "AUDIO_CHUNK_DURATION": 2.0,
    "CONFIDENCE_THRESHOLD_CONFLICT": 0.6,
    "CONFIDENCE_DIFF_LOGGING": 0.15,
    "FALLBACK_CONFIDENCE": 0.4,
    "UI_WIDTH": 1200,
    "UI_HEIGHT": 700
}

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, 'r') as f:
            user_config = json.load(f)
            CONFIG.update(user_config)
            print(f"Loaded configuration from {CONFIG_FILE}")
    except Exception as e:
        print(f"Error loading config.json: {e}")

# --- Apply Configuration ---
WEBCAM_ID = CONFIG["WEBCAM_ID"]
FACE_DETECTION_THRESHOLD = CONFIG["FACE_DETECTION_THRESHOLD"]
INPUT_IMAGE_SIZE = 224 # Fixed for EfficientNet
VISUAL_INFERENCE_INTERVAL = CONFIG["VISUAL_INFERENCE_INTERVAL"]

# Normalization (ImageNet defaults)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Audio Configuration ---
SAMPLE_RATE = 16000
AUDIO_CHUNK_DURATION = 3.0 # Updated for ResNet-18 (was 2.0)
AUDIO_CHANNELS = 1
N_MFCC = 128 # Updated for Log-Mel Spectrogram (was 40)
AUDIO_INFERENCE_INTERVAL = 0.5 

# --- Fusion Configuration ---
CONFIDENCE_THRESHOLD_CONFLICT = CONFIG["CONFIDENCE_THRESHOLD_CONFLICT"]
CONFIDENCE_DIFF_LOGGING = CONFIG["CONFIDENCE_DIFF_LOGGING"]
FALLBACK_CONFIDENCE = CONFIG["FALLBACK_CONFIDENCE"]

# --- UI Configuration ---
WINDOW_NAME = "MERS - Real-Time Multimodal Emotion Recognition"
UI_WIDTH = CONFIG["UI_WIDTH"]
UI_HEIGHT = CONFIG["UI_HEIGHT"]
BG_COLOR = (20, 20, 20)
TEXT_COLOR = (255, 255, 255)
ACCENT_COLOR = (0, 255, 200)

# --- Visual Identity (Colors) ---
# Format: (B, G, R) for OpenCV, Hex for Web/Flutter
EMOTION_COLORS = {
    "Happy":    {"hex": "#2ECC71", "bgr": (113, 204, 46)},   # Green
    "Sad":      {"hex": "#3498DB", "bgr": (219, 152, 52)},   # Blue
    "Angry":    {"hex": "#E74C3C", "bgr": (60, 76, 231)},    # Red
    "Fear":     {"hex": "#9B59B6", "bgr": (182, 89, 155)},   # Purple
    "Neutral":  {"hex": "#95A5A6", "bgr": (166, 165, 149)},  # Gray
    "Surprise": {"hex": "#F1C40F", "bgr": (15, 196, 241)},   # Yellow/Orange
    "Disgust":  {"hex": "#E67E22", "bgr": (34, 126, 230)}    # Orange
}

# --- Visual Identity (Emojis) ---
EMOTION_EMOJIS = {
    "Happy":    "😊",
    "Sad":      "😢",
    "Angry":    "😠",
    "Fear":     "😨",
    "Neutral":  "😐",
    "Surprise": "😲",
    "Disgust":  "🤢"
}

def get_color(emotion, format="bgr"):
    """
    Returns color for emotion. 
    Defaults to Gray/Neutral if not found.
    """
    default = EMOTION_COLORS["Neutral"]
    color_data = EMOTION_COLORS.get(emotion, default)
    return color_data.get(format, default["bgr"])
