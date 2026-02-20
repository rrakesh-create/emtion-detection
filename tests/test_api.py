import requests
import numpy as np
import scipy.io.wavfile as wav
import os

# Configuration
BASE_URL = "http://127.0.0.1:8000"
AUDIO_FILE = "test_audio.wav"

def generate_dummy_wav():
    print("Generating dummy WAV file...")
    sample_rate = 16000
    duration = 3  # seconds
    frequency = 440  # Hz
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate a sine wave
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    wav.write(AUDIO_FILE, sample_rate, audio_data)
    print(f"Created {AUDIO_FILE}")

def test_health():
    print("\n[Testing /health]...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Failed: {e}")

def test_analyze_audio():
    print("\n[Testing /analyze_audio]...")
    try:
        with open(AUDIO_FILE, "rb") as f:
            files = {"file": ("test_audio.wav", f, "audio/wav")}
            response = requests.post(f"{BASE_URL}/analyze_audio", files=files)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.json()}")
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed: {e}")

def test_analyze_multimodal():
    print("\n[Testing /analyze_multimodal]...")
    try:
        with open(AUDIO_FILE, "rb") as f:
            files = {"file": ("test_audio.wav", f, "audio/wav")}
            response = requests.post(f"{BASE_URL}/analyze_multimodal", files=files)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.json()}")
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    generate_dummy_wav()
    test_health()
    test_analyze_audio()
    test_analyze_multimodal()
    
    # Cleanup
    # os.remove(AUDIO_FILE)
