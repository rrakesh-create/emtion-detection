import os
import sys
import queue
from typing import Optional, Dict, Tuple, Any

import numpy as np
import librosa
import sounddevice as sd
import joblib
import torch
import torch.nn as nn
from torchvision import models

from config.settings import SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_CHUNK_DURATION, EMOTIONS, MODEL_EMOTIONS, MODELS_DIR, N_MFCC
# from mers.core.features import extract_audio_features # Deprecated for ResNet

class AudioEngine:
    def __init__(self):
        print(f"[AudioEngine] Initializing (Deep Learning ResNet-18 mode)...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AudioEngine] Using device: {self.device}")

        # Resolve model paths
        self.model_path = os.path.join(MODELS_DIR, "audio_resnet18.pth")
        self.encoder_path = os.path.join(MODELS_DIR, "label_encoder_cnn.pkl")
        
        self.model = None
        self.le = None
        
        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            try:
                print(f"[AudioEngine] Loading ResNet-18 from {self.model_path}")
                
                # Reconstruct Model Architecture
                self.model = models.resnet18(pretrained=False) # No need to downlaod weights, we load ours
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                num_ftrs = self.model.fc.in_features
                
                # Load Encoder to know num_classes
                self.le = joblib.load(self.encoder_path)
                num_classes = len(self.le.classes_)
                self.model.fc = nn.Linear(num_ftrs, num_classes)
                
                # Load Weights
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                
                print(f"[AudioEngine] Model loaded successfully. Classes: {self.le.classes_}")
                
            except Exception as e:
                print(f"[AudioEngine] Error loading model: {e}")
                self.model = None
        else:
            print("[AudioEngine] Warning: ResNet Model/Encoder not found. Using fallback.")

        self.audio_queue = queue.Queue()
        self.running = False
        
        # Buffer for live stream
        self.buffer = np.zeros((int(SAMPLE_RATE * AUDIO_CHUNK_DURATION), 1))
        
        # Flags
        self.enable_noise_suppression = False
        self.enable_silence_trimming = False

    def start_stream(self):
        self.running = True
        self.stream = sd.InputStream(
            channels=AUDIO_CHANNELS,
            samplerate=SAMPLE_RATE,
            callback=self._audio_callback,
            blocksize=int(SAMPLE_RATE * 0.5)
        )
        self.stream.start()

    def stop_stream(self):
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def process_queue(self):
        """
        For real-time local mic.
        """
        if self.audio_queue.empty():
            return None, 0.0, None

        while not self.audio_queue.empty():
            new_data = self.audio_queue.get()
            chunk_len = len(new_data)
            self.buffer = np.roll(self.buffer, -chunk_len, axis=0)
            self.buffer[-chunk_len:] = new_data
        
        buffer_flat = self.buffer.flatten()
        return self.process_audio_data(buffer_flat)

    def _preprocess_audio(self, audio_data):
        """
        Convert audio chunk to Normalized Log-Mel Spectrogram Tensor
        """
        # Ensure correct length
        target_len = int(SAMPLE_RATE * AUDIO_CHUNK_DURATION)
        if len(audio_data) > target_len:
            audio_data = audio_data[:target_len]
        else:
            pad_width = target_len - len(audio_data)
            audio_data = np.pad(audio_data, (0, pad_width), mode='constant')
            
        # Compute Log-Mel Spectrogram
        # Note: N_MFCC here represents n_mels (128)
        melspec = librosa.feature.melspectrogram(y=audio_data, sr=SAMPLE_RATE, n_mels=N_MFCC, hop_length=512)
        melspec = librosa.power_to_db(melspec, ref=np.max)
        
        # Normalize
        mean = np.mean(melspec)
        std = np.std(melspec)
        if std > 0:
            melspec = (melspec - mean) / std
        else:
            melspec = (melspec - mean)

        # To Tensor: (1, n_mels, time_steps)
        tensor = torch.tensor(melspec, dtype=torch.float32).unsqueeze(0) # Add channel dim
        tensor = tensor.unsqueeze(0) # Add batch dim: (1, 1, H, W)
        return tensor

    def process_audio_data(self, audio_data: np.ndarray):
        """
        Unified processing for both Stream (Mic) and API (File).
        Returns:
            - probs: np.array (prediction in EMOTIONS order)
            - rms: float
            - features: Dict (for XAI)
        """
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Check silence
        if rms < 0.001:
            return None, rms, {}

        features_xai = {"energy_mean": rms}

        try:
            # 1. Inference
            if self.model is not None:
                # Preprocess
                input_tensor = self._preprocess_audio(audio_data).to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probs_raw = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                # Map from Model Classes to Output EMOTIONS
                # self.le.classes_ gives specific order (e.g. ['Angry', 'Happy'...])
                # We need to map this to EMOTIONS list order
                
                model_classes = self.le.classes_
                probs = np.zeros(len(EMOTIONS))
                
                top_idx = np.argmax(probs_raw)
                top_label = model_classes[top_idx]
                
                features_xai["top_detected"] = top_label
                
                for i, class_name in enumerate(model_classes):
                    # Map class_name (e.g. "Angry") to EMOTIONS index
                    if class_name in EMOTIONS:
                        out_idx = EMOTIONS.index(class_name)
                        probs[out_idx] = probs_raw[i]
                    elif class_name == "nuetral": # Handle typo in dataset if present
                         if "Neutral" in EMOTIONS:
                            probs[EMOTIONS.index("Neutral")] = probs_raw[i]
                            
            else:
                # Fallback if no model
                probs = np.zeros(len(EMOTIONS))
                if "Neutral" in EMOTIONS:
                    probs[EMOTIONS.index("Neutral")] = 1.0
                
        except Exception as e:
            print(f"[AudioEngine] Inference Error: {e}")
            return None, rms, {}
            
        return probs, rms, features_xai
