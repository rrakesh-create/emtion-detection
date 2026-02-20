import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import numpy as np
import threading
import os
import json

# Configuration
SERVER_URL = "http://127.0.0.1:8000"
SAMPLE_RATE = 16000
CHANNELS = 1
FILENAME = "client_record.wav"

class MERSClientApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MERS PC Client")
        self.root.geometry("400x500")
        
        # Styles
        self.style = ttk.Style()
        self.style.configure("TButton", padding=10, font=('Helvetica', 12))
        self.style.configure("TLabel", font=('Helvetica', 11))
        
        # Title
        ttk.Label(root, text="Multimodal Emotion Recognition", font=('Helvetica', 16, 'bold')).pack(pady=20)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(root, textvariable=self.status_var, foreground="blue")
        self.status_label.pack(pady=10)
        
        # Record Button
        self.record_btn = ttk.Button(root, text="Hold to Record", command=self.start_recording)
        self.record_btn.bind('<ButtonPress-1>', self.start_recording)
        self.record_btn.bind('<ButtonRelease-1>', self.stop_recording)
        self.record_btn.pack(pady=20)
        
        # Results Frame
        self.result_frame = ttk.LabelFrame(root, text="Analysis Result", padding=15)
        self.result_frame.pack(fill="x", padx=20, pady=10)
        
        self.emotion_var = tk.StringVar(value="--")
        self.confidence_var = tk.StringVar(value="--")
        self.explanation_var = tk.StringVar(value="--")
        
        self.create_result_row("Emotion:", self.emotion_var)
        self.create_result_row("Confidence:", self.confidence_var)
        self.create_result_row("Explanation:", self.explanation_var, wraplength=320)
        
        self.is_recording = False
        self.recording_thread = None
        self.audio_data = []

    def create_result_row(self, label, variable, wraplength=None):
        frame = ttk.Frame(self.result_frame)
        frame.pack(fill="x", pady=5)
        ttk.Label(frame, text=label, width=12, font=('Helvetica', 10, 'bold')).pack(side="left", anchor="n")
        lbl = ttk.Label(frame, textvariable=variable, wraplength=wraplength)
        lbl.pack(side="left", fill="x", expand=True)

    def start_recording(self, event=None):
        if self.is_recording: return
        self.is_recording = True
        self.status_var.set("Recording...")
        self.audio_data = []
        self.recording_thread = threading.Thread(target=self._record_loop)
        self.recording_thread.start()

    def stop_recording(self, event=None):
        if not self.is_recording: return
        self.is_recording = False
        self.status_var.set("Processing...")
        
    def _record_loop(self):
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16') as stream:
            while self.is_recording:
                data, overflowed = stream.read(1024)
                self.audio_data.append(data)
        
        # Save file
        if self.audio_data:
            full_recording = np.concatenate(self.audio_data, axis=0)
            wav.write(FILENAME, SAMPLE_RATE, full_recording)
            self.send_audio(FILENAME)

    def send_audio(self, filename):
        try:
            with open(filename, 'rb') as f:
                files = {'file': (filename, f, 'audio/wav')}
                response = requests.post(f"{SERVER_URL}/analyze_audio", files=files)
            
            if response.status_code == 200:
                result = response.json()
                self.root.after(0, self.update_ui, result)
            else:
                self.root.after(0, lambda: self.status_var.set(f"Error: {response.text}"))
                
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Connection Error: {e}"))

    def update_ui(self, result):
        self.status_var.set("Analysis Complete")
        self.emotion_var.set(result.get('emotion', 'Unknown'))
        conf = result.get('confidence', 0)
        self.confidence_var.set(f"{conf:.2f}")
        self.explanation_var.set(result.get('explanation', 'No explanation'))

if __name__ == "__main__":
    root = tk.Tk()
    app = MERSClientApp(root)
    root.mainloop()
