import warnings
from typing import Dict, Tuple, Any, List

import numpy as np
import librosa

def extract_audio_features(y: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extracts comprehensive prosody and spectral features for high-accuracy ML models.
    
    Features (Total ~85 dimensions):
    - Energy (RMS): Mean, Std
    - Zero Crossing Rate: Mean
    - Spectral Centroid: Mean, Std
    - Spectral Rolloff: Mean
    - Spectral Contrast: Mean (7 bands)
    - MFCCs: Mean (20 coeffs)
    - Chroma STFT: Mean (12 bins)
    - Tonnetz: Mean (6 dims)
    - Mel Spectrogram: Mean (scaled)
    - Speaking Rate
    """
    warnings.filterwarnings("ignore")
    features = {}
    
    # 1. Energy (RMS)
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # 2. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = np.mean(zcr)
    
    # 3. Spectral Features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['centroid_mean'] = np.mean(centroid)
    features['centroid_std'] = np.std(centroid)
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['rolloff_mean'] = np.mean(rolloff)
    
    # Spectral Contrast (7 bands)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    for i, val in enumerate(contrast_mean):
        features[f'contrast_{i}'] = val
        
    # 4. MFCCs (Increased to 20 for better detail)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfccs, axis=1)
    for i, val in enumerate(mfcc_mean):
        features[f'mfcc_{i+1}'] = val
        
    # 5. Chroma & Tonnetz (Harmonic/Pitch content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    for i, val in enumerate(chroma_mean):
        features[f'chroma_{i+1}'] = val
        
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    for i, val in enumerate(tonnetz_mean):
        features[f'tonnetz_{i+1}'] = val
        
    # 6. Mel Spectrogram (Texture)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    features['mel_mean'] = np.mean(librosa.power_to_db(mel))

    # 7. Speaking Rate (Approximate)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    duration = len(y) / sr
    features['speaking_rate'] = len(onsets) / (duration + 1e-6)
    
    # Flatten into vector (Order must be consistent!)
    # We construct a list of values sorted by key to ensure consistency or manual list
    # Manual list is safer for strict ordering
    
    vector_list = [
        features['rms_mean'], features['rms_std'],
        features['zcr_mean'],
        features['centroid_mean'], features['centroid_std'],
        features['rolloff_mean'],
        features['mel_mean'],
        features['speaking_rate']
    ]
    
    # Append Contrast (7)
    vector_list.extend([features[f'contrast_{i}'] for i in range(7)])
    # Append MFCC (20)
    vector_list.extend([features[f'mfcc_{i+1}'] for i in range(20)])
    # Append Chroma (12)
    vector_list.extend([features[f'chroma_{i+1}'] for i in range(12)])
    # Append Tonnetz (6)
    vector_list.extend([features[f'tonnetz_{i+1}'] for i in range(6)])
    
    return np.array(vector_list), features

def extract_visual_features(landmarks):
    """
    Extracts geometric features from 468 MediaPipe landmarks.
    Input: landmarks (468, 3) or list of objects with .x, .y
    """
    # Placeholder for geometric feature extraction
    # Distances between key points (lips, eyes, eyebrows)
    # For now, we return raw flattened landmarks or a subset if we were training.
    # Since we aren't training visual yet, this is structural.
    return np.array(landmarks).flatten()
