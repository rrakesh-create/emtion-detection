import os
import threading
import queue
import time
import uuid
from collections import deque
import tkinter as tk
from tkinter import ttk, Canvas, messagebox
# Matplotlib for Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import cv2
import numpy as np
import sounddevice as sd

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
try:
    from mediapipe import solutions
    mp_drawing = solutions.drawing_utils
    mp_face_mesh = solutions.face_mesh
except ImportError:
    print("[Dashboard] Warning: Could not import mediapipe.solutions. Mesh drawing disabled.")
    mp_drawing = None
    mp_face_mesh = None
from PIL import Image, ImageTk

from mers.core.visual_engine import VisualEngine
from mers.core.audio_engine import AudioEngine
from mers.core.fusion_engine import FusionEngine
from mers.core.xai_engine import XAIEngine
from config.settings import EMOTION_COLORS, EMOTION_EMOJIS, EMOTIONS, MODELS_DIR, LOG_FILE

# Conditional Import for Visual Engine (CNN vs Rules)
try:
    from mers.core.visual_engine_cnn import VisualEngineCNN
    HAS_CNN = True
except ImportError:
    HAS_CNN = False

SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 1024
AUDIO_WINDOW_SECONDS = 3
WAVEFORM_POINTS = 200

import datetime

class ScrollableFrame(tk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, bg=kwargs.get("bg", "#252526"), highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        self.scrollable_frame = tk.Frame(self.canvas, bg=kwargs.get("bg", "#252526"))
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Adjust inner frame width on resize
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Mousewheel
        self.bind("<Enter>", self._bind_mouse)
        self.bind("<Leave>", self._unbind_mouse)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_frame, width=event.width)

    def _bind_mouse(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
    def _unbind_mouse(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

class MERSDashboard:
    def __init__(self, root):
        # Prevent Server from grabbing camera
        os.environ["SKIP_SERVER_CAMERA"] = "true"
        
        self.root = root
        self.root.title("MERS - Multimodal Emotion Recognition System")
        self.root.geometry("1600x900") # Wider for 3-column layout
        self.root.configure(bg="#1e1e1e")

        # Session ID
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.datetime.now()

        # Initialize Logging
        self.log_file = LOG_FILE
        # Directory is already created by settings.py, but safe to ensure
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(f"\n--- Session Started: {self.session_start} ---\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write("Timestamp | Visual Emotion (Conf) | Audio Emotion (Conf) | Fused Emotion (Conf) | Why\n")

        # Visual Options
        self.show_mesh_var = tk.BooleanVar(value=False)
        self.show_border_var = tk.BooleanVar(value=True)
        
        # Control Vars
        self.cam_enabled_var = tk.BooleanVar(value=True)
        self.fps_var = tk.IntVar(value=30)
        self.resolution_var = tk.StringVar(value="640x480")
        self.conf_threshold_var = tk.DoubleVar(value=0.5)
        
        self.noise_supp_var = tk.BooleanVar(value=False)
        self.silence_trim_var = tk.BooleanVar(value=False)
        
        self.visual_weight_var = tk.DoubleVar(value=0.5)
        self.audio_weight_var = tk.DoubleVar(value=0.5)
        self.enable_visual_fusion = tk.BooleanVar(value=True)
        self.enable_audio_fusion = tk.BooleanVar(value=True)

        # System Options
        self.auto_stop_no_face_var = tk.BooleanVar(value=False)
        self.auto_stop_silence_var = tk.BooleanVar(value=False)
        self.error_logging_var = tk.BooleanVar(value=True)
        
        # State Timers
        self.no_face_start_time = None
        self.silence_start_time = None

        # Initialize Engines
        print("[Dashboard] Initializing Engines...")
        
        # Check if CNN model exists and load it, otherwise fallback to rules
        model_path = os.path.join(MODELS_DIR, "visual_efficientnet.pth")
        if HAS_CNN and os.path.exists(model_path):
            print("[Dashboard] Found CNN model. Using VisualEngineCNN.")
            self.visual_engine = VisualEngineCNN(model_path=model_path)
        else:
            print("[Dashboard] CNN model not found. Using Rule-Based VisualEngine.")
            self.visual_engine = VisualEngine()
            
        self.audio_engine = AudioEngine()
        self.fusion_engine = FusionEngine()
        self.xai_engine = XAIEngine()
        
        # Database Manager
        from mers.core.db_manager import DatabaseManager
        self.db_manager = DatabaseManager()
        self.db_manager.create_session(self.session_id)
        
        # Audio State
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE * AUDIO_WINDOW_SECONDS)) # 3 seconds rolling buffer
        self.waveform_data = deque([0]*WAVEFORM_POINTS, maxlen=WAVEFORM_POINTS)
        self.is_running = True
        
        # Historical Data for Graphs
        self.history_len = 50
        self.conf_history = deque([0.0]*self.history_len, maxlen=self.history_len)
        self.audio_conf_history = deque([0.0]*self.history_len, maxlen=self.history_len)
        self.visual_conf_history = deque([0.0]*self.history_len, maxlen=self.history_len)
        
        # UI Setup
        self._setup_ui()
        
        # Start Threads
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        
        self.video_thread.start()
        self.audio_thread.start()
        self.analysis_thread.start()

        # Update Timer
        self.root.after(50, self._update_ui)

    def _setup_ui(self):
        # Create Notebook (Tabs)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background="#1e1e1e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#2d2d2d", foreground="white", padding=[10, 5], font=('Helvetica', 10))
        style.map("TNotebook.Tab", background=[("selected", "#4a90e2")], foreground=[("selected", "white")])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Live Analysis
        self.live_tab = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.live_tab, text="   🎥 Live Analysis   ")
        
        # Tab 2: History & Trends
        self.history_tab = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.history_tab, text="   📊 Mood History   ")
        
        # --- Build Live Tab ---
        container = self.live_tab # Bind existing panel logic to this frame
        
        # 1. Left: Video (Dynamic)
        self.video_frame = tk.Frame(container, bg="black")
        self.video_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)

        # 2. Center: Analysis (Fixed)
        self.center_panel = tk.Frame(container, bg="#2d2d2d", width=350)
        self.center_panel.pack(side="left", fill="y", padx=5, pady=5)
        self.center_panel.pack_propagate(False)
        self._build_analysis_panel(self.center_panel)

        # 3. Right: Controls (Scrollable)
        self.right_panel = ScrollableFrame(container, bg="#252526", width=320)
        self.right_panel.pack(side="left", fill="y", padx=5, pady=5)
        self.right_panel.pack_propagate(False)
        self._build_controls_panel(self.right_panel.scrollable_frame)
        
        # --- Build History Tab ---
        self._build_history_tab()

    def _build_history_tab(self):
        # Top Bar: Refresh Button
        top_bar = tk.Frame(self.history_tab, bg="#252526", height=50)
        top_bar.pack(fill="x", side="top")
        
        tk.Button(top_bar, text="🔄 Refresh Data", command=self._refresh_history, bg="#4a90e2", fg="white", font=("Arial", 10, "bold"), padx=15, pady=5, borderwidth=0).pack(pady=10)
        
        # Content Grid
        content = tk.Frame(self.history_tab, bg="#1e1e1e")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Left: Chart
        self.chart_frame = tk.Frame(content, bg="#2d2d2d")
        self.chart_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        tk.Label(self.chart_frame, text="Session Emotion Distribution", font=("Helvetica", 12, "bold"), fg="white", bg="#2d2d2d").pack(pady=10)
        
        # Matplotlib Figure
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#2d2d2d')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#2d2d2d')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        for spine in self.ax.spines.values():
            spine.set_color('white')
            
        self.canvas_chart = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas_chart.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Right: Recent Sessions Table
        table_frame = tk.Frame(content, bg="#2d2d2d", width=400)
        table_frame.pack(side="right", fill="y")
        table_frame.pack_propagate(False)
        
        tk.Label(table_frame, text="Recent Sessions", font=("Helvetica", 12, "bold"), fg="white", bg="#2d2d2d").pack(pady=10)
        
        # Treeview Style
        style = ttk.Style()
        style.configure("Treeview", background="#333333", foreground="white", fieldbackground="#333333", rowheight=25)
        style.configure("Treeview.Heading", background="#444444", foreground="black", font=('Arial', 10, 'bold'))
        
        columns = ("start_time", "emotion")
        self.history_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        self.history_tree.heading("start_time", text="Date/Time")
        self.history_tree.heading("emotion", text="Dominant Emotion")
        
        self.history_tree.column("start_time", width=220)
        self.history_tree.column("emotion", width=120)
        
        self.history_tree.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initial Load
        self._refresh_history()

    def _refresh_history(self):
        # Clear Table
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
            
        # Clear Chart
        self.ax.clear()
        
        # Fetch Data
        sessions = self.db_manager.get_session_history_all()
        
        if not sessions:
            return

        # Populate Table
        emotion_counts = {}
        for s in sessions: # Limit to last 50 for table efficiency? No, allow scroll.
            # s is a sqlite3.Row object, access by key
            t_str = s['start_time'][:19]
            emo = s['dominant_emotion'] or "Unknown"
            self.history_tree.insert("", "end", values=(t_str, emo))
            
            if emo not in ["None", "Unknown"]:
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
        
        # Draw Chart
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        
        # Colors
        colors = []
        for e in emotions:
            # Try to match config colors
            if e in EMOTION_COLORS:
                # Hex needed for matplotlib
                colors.append(EMOTION_COLORS[e].get('hex', '#888888'))
            else:
                colors.append('#888888')
                
        bars = self.ax.bar(emotions, counts, color=colors)
        self.ax.set_title("Sessions by Dominant Emotion", color='white')
        self.ax.set_ylabel("Count", color='white')
        
        self.canvas_chart.draw()

    def _build_analysis_panel(self, parent):
        # Header
        tk.Label(parent, text="Real-Time Analysis", font=("Helvetica", 16, "bold"), fg="white", bg="#2d2d2d").pack(pady=10)
        
        # Primary Emotion Card
        self.card_frame = tk.Frame(parent, bg="#333333", padx=10, pady=10)
        self.card_frame.pack(fill="x", padx=10, pady=10)
        self.emotion_label = tk.Label(self.card_frame, text="--", font=("Helvetica", 28, "bold"), fg="#ffffff", bg="#333333")
        self.emotion_label.pack()
        self.confidence_label = tk.Label(self.card_frame, text="Confidence: --%", font=("Helvetica", 10), fg="#aaaaaa", bg="#333333")
        self.confidence_label.pack()

        # Confidence Bar
        self.confidence_bar = ttk.Progressbar(parent, orient="horizontal", length=280, mode="determinate")
        self.confidence_bar.pack(pady=5)

        # Modality Contribution
        tk.Label(parent, text="Modality Contribution", font=("Helvetica", 11, "bold"), fg="white", bg="#2d2d2d").pack(pady=(15, 5), anchor="w", padx=10)
        self.modality_frame = tk.Frame(parent, bg="#2d2d2d")
        self.modality_frame.pack(fill="x", padx=10)
        
        self.audio_contrib_label = tk.Label(self.modality_frame, text="Audio: 0%", fg="white", bg="#2d2d2d", font=("Helvetica", 9))
        self.audio_contrib_label.grid(row=0, column=0, sticky="w")
        self.audio_contrib_bar = ttk.Progressbar(self.modality_frame, orient="horizontal", length=180, mode="determinate")
        self.audio_contrib_bar.grid(row=0, column=1, padx=5)
        
        self.visual_contrib_label = tk.Label(self.modality_frame, text="Visual: 0%", fg="white", bg="#2d2d2d", font=("Helvetica", 9))
        self.visual_contrib_label.grid(row=1, column=0, sticky="w")
        self.visual_contrib_bar = ttk.Progressbar(self.modality_frame, orient="horizontal", length=180, mode="determinate")
        self.visual_contrib_bar.grid(row=1, column=1, padx=5)

        # Explanation
        tk.Label(parent, text="Why?", font=("Helvetica", 12, "bold"), fg="white", bg="#2d2d2d").pack(pady=(15, 5), anchor="w", padx=10)
        self.expl_frame = tk.Frame(parent, bg="#333333", padx=5, pady=5)
        self.expl_frame.pack(fill="x", padx=10)
        self.visual_expl = tk.Label(self.expl_frame, text="...", font=("Helvetica", 9), fg="#dddddd", bg="#333333", wraplength=300, justify="left")
        self.visual_expl.pack(anchor="w", pady=2)
        self.audio_expl = tk.Label(self.expl_frame, text="...", font=("Helvetica", 9), fg="#dddddd", bg="#333333", wraplength=300, justify="left")
        self.audio_expl.pack(anchor="w", pady=2)

        # Graphs
        tk.Label(parent, text="Confidence History", font=("Helvetica", 11, "bold"), fg="white", bg="#2d2d2d").pack(pady=(15, 5), anchor="w", padx=10)
        self.graph_canvas = Canvas(parent, bg="black", height=80, width=320, highlightthickness=0)
        self.graph_canvas.pack(pady=5)

        tk.Label(parent, text="Audio Waveform", font=("Helvetica", 11, "bold"), fg="white", bg="#2d2d2d").pack(pady=(10, 5), anchor="w", padx=10)
        self.waveform_canvas = Canvas(parent, bg="black", height=50, width=320, highlightthickness=0)
        self.waveform_canvas.pack(pady=5)

    def _build_controls_panel(self, parent):
        tk.Label(parent, text="Controls & Settings", font=("Helvetica", 16, "bold"), fg="white", bg="#252526").pack(pady=10)
        
        def add_section(title):
            tk.Label(parent, text=title, font=("Helvetica", 11, "bold"), fg="#4EC9B0", bg="#252526").pack(pady=(15, 5), anchor="w", padx=10)
            f = tk.Frame(parent, bg="#252526")
            f.pack(fill="x", padx=15)
            return f

        # Video Controls
        v_frame = add_section("Video Input")
        tk.Checkbutton(v_frame, text="Enable Camera", variable=self.cam_enabled_var, bg="#252526", fg="white", selectcolor="#444444", activebackground="#252526").pack(anchor="w")
        
        f_res = tk.Frame(v_frame, bg="#252526")
        f_res.pack(fill="x", pady=2)
        tk.Label(f_res, text="Resolution:", fg="white", bg="#252526").pack(side="left")
        ttk.Combobox(f_res, textvariable=self.resolution_var, values=["640x480", "1280x720"], width=10).pack(side="right")
        
        f_fps = tk.Frame(v_frame, bg="#252526")
        f_fps.pack(fill="x", pady=2)
        tk.Label(f_fps, text="FPS Limit:", fg="white", bg="#252526").pack(side="left")
        ttk.Combobox(f_fps, textvariable=self.fps_var, values=[5, 10, 15, 30, 60], width=5).pack(side="right")

        tk.Label(v_frame, text="Face Conf. Threshold:", fg="white", bg="#252526").pack(anchor="w", pady=(5,0))
        tk.Scale(v_frame, variable=self.conf_threshold_var, from_=0.1, to=1.0, resolution=0.05, orient="horizontal", bg="#252526", fg="white", highlightthickness=0).pack(fill="x")

        # Audio Controls
        a_frame = add_section("Audio Input")
        tk.Checkbutton(a_frame, text="Noise Suppression", variable=self.noise_supp_var, command=self._apply_audio_settings, bg="#252526", fg="white", selectcolor="#444444", activebackground="#252526").pack(anchor="w")
        tk.Checkbutton(a_frame, text="Silence Trimming", variable=self.silence_trim_var, command=self._apply_audio_settings, bg="#252526", fg="white", selectcolor="#444444", activebackground="#252526").pack(anchor="w")

        # Fusion Controls
        f_frame = add_section("Multimodal Fusion")
        tk.Checkbutton(f_frame, text="Enable Visual", variable=self.enable_visual_fusion, command=self._apply_fusion_settings, bg="#252526", fg="white", selectcolor="#444444", activebackground="#252526").pack(anchor="w")
        tk.Checkbutton(f_frame, text="Enable Audio", variable=self.enable_audio_fusion, command=self._apply_fusion_settings, bg="#252526", fg="white", selectcolor="#444444", activebackground="#252526").pack(anchor="w")
        
        tk.Label(f_frame, text="Visual Weight:", fg="#aaaaaa", bg="#252526", font=("Arial", 8)).pack(anchor="w")
        tk.Scale(f_frame, variable=self.visual_weight_var, from_=0.0, to=1.0, resolution=0.1, orient="horizontal", bg="#252526", fg="white", highlightthickness=0, command=lambda x: self._apply_fusion_settings()).pack(fill="x")
        
        tk.Label(f_frame, text="Audio Weight:", fg="#aaaaaa", bg="#252526", font=("Arial", 8)).pack(anchor="w")
        tk.Scale(f_frame, variable=self.audio_weight_var, from_=0.0, to=1.0, resolution=0.1, orient="horizontal", bg="#252526", fg="white", highlightthickness=0, command=lambda x: self._apply_fusion_settings()).pack(fill="x")

        # Visualization
        v_ctrl = add_section("Visualization")
        tk.Checkbutton(v_ctrl, text="Show Face Box", variable=self.show_border_var, bg="#252526", fg="white", selectcolor="#444444", activebackground="#252526").pack(anchor="w")
        tk.Checkbutton(v_ctrl, text="Show Face Mesh", variable=self.show_mesh_var, bg="#252526", fg="white", selectcolor="#444444", activebackground="#252526").pack(anchor="w")

        # Session
        s_frame = add_section("Session Info")
        tk.Label(s_frame, text=f"ID: {self.session_id[:8]}...", fg="#888888", bg="#252526").pack(anchor="w")

        # System Controls
        sys_frame = add_section("System & Safety")
        tk.Checkbutton(sys_frame, text="Auto-Stop (No Face)", variable=self.auto_stop_no_face_var, bg="#252526", fg="white", selectcolor="#444444", activebackground="#252526").pack(anchor="w")
        tk.Checkbutton(sys_frame, text="Auto-Stop (Silence)", variable=self.auto_stop_silence_var, bg="#252526", fg="white", selectcolor="#444444", activebackground="#252526").pack(anchor="w")
        tk.Checkbutton(sys_frame, text="Error Logging", variable=self.error_logging_var, bg="#252526", fg="white", selectcolor="#444444", activebackground="#252526").pack(anchor="w")

    def _apply_audio_settings(self):
        self.audio_engine.enable_noise_suppression = self.noise_supp_var.get()
        self.audio_engine.enable_silence_trimming = self.silence_trim_var.get()

    def _apply_fusion_settings(self):
        disabled = []
        if not self.enable_visual_fusion.get(): disabled.append("visual")
        if not self.enable_audio_fusion.get(): disabled.append("audio")
        
        weights = {
            "visual": self.visual_weight_var.get(),
            "audio": self.audio_weight_var.get()
        }
        self.fusion_engine.set_config(weights=weights, disabled=disabled)


    def _video_loop(self):
        print("[Dashboard] Attempting to open camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
             print("[Dashboard] ERROR: Could not open camera (index 0). Trying index 1...")
             cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
             if not cap.isOpened():
                 print("[Dashboard] FATAL: Could not open any camera.")
                 return

        print("[Dashboard] Camera opened successfully.")
        
        while self.is_running:
            ret, frame = cap.read()
            if ret:
                try:
                    # 1. Process Visual Emotion
                    # visual_engine now returns landmarks as 4th element
                    visual_probs, face_box, visual_features, landmarks = self.visual_engine.process_frame(frame)
                    
                    # Auto-Stop Logic (No Face)
                    if visual_probs is None:
                        if self.no_face_start_time is None:
                            self.no_face_start_time = time.time()
                        elif time.time() - self.no_face_start_time > 5.0 and self.auto_stop_no_face_var.get():
                            print("[System] Auto-Stop: No face detected for 5s.")
                            self.is_running = False
                            break
                    else:
                        self.no_face_start_time = None

                    # Determine current emotion color for box
                    box_color = (255, 255, 0) # Default Teal (BGR)
                    if hasattr(self, 'final_result'):
                        emo = self.final_result.get('emotion', 'Neutral')
                        if emo in EMOTION_COLORS:
                            # Config uses RGB tuple or Hex. We need BGR for OpenCV.
                            # EMOTION_COLORS[emo]["bgr"] is already (B, G, R)? Check config.
                            # Config says: "bgr": (113, 204, 46) which looks like RGB actually in previous snippet?
                            # Let's assume the config has BGR or we convert.
                            # Snippet in memory: "Happy": {"hex": "#2ECC71", "bgr": (113, 204, 46)}
                            # #2ECC71 is Green. (R=46, G=204, B=113).
                            # So (113, 204, 46) is (B, G, R). Correct.
                            box_color = EMOTION_COLORS[emo]["bgr"]

                    if landmarks:
                        # Draw Face Mesh (Mapping)
                        if self.show_mesh_var.get():
                            # We need to construct a Mock result for draw_landmarks or manually draw points
                            # landmarks is a list of NormalizedLandmark.
                            # mp_drawing.draw_landmarks expects a NormalizedLandmarkList proto usually.
                            # But we can reconstruct it or just draw points manually for speed.
                            # Actually, VisualEngine returns the list from detection_result.face_landmarks[0]
                            # We can wrap it in a class to satisfy mp_drawing or just loop.
                            h, w, c = frame.shape
                            for lm in landmarks:
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(frame, (cx, cy), 1, (255, 255, 255), -1)
                            
                            # Draw contours (Eyes, Lips) for better effect?
                            # Simple dots is fine for "Mesh" request.

                    if face_box and self.show_border_var.get():
                        x1, y1, x2, y2 = face_box
                        # Draw dynamic border with emotion color
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                        
                        # Label
                        if hasattr(self, 'final_result'):
                            txt = f"{self.final_result.get('emotion', '')} {int(self.final_result.get('confidence',0)*100)}%"
                            cv2.putText(frame, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                        
                    self.latest_visual_result = {
                        'probabilities': visual_probs,
                        'landmarks': visual_features
                    }
                    
                    # UPDATE SERVER GLOBAL VARIABLE FOR MOBILE API

                    
                    # Convert BGR (OpenCV) -> RGB (Tkinter)
                    frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb_display)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    self.current_image = imgtk
                    
                except Exception as e:
                    if self.error_logging_var.get():
                        print(f"[Dashboard] Error in video loop: {e}")
                    time.sleep(0.1)
            else:
                if self.error_logging_var.get():
                    print("[Dashboard] Warning: Empty frame from camera.")
                time.sleep(0.1)
        cap.release()

    def _audio_loop(self):
        def callback(indata, frames, time, status):
            if status and self.error_logging_var.get():
                print(status)
            self.audio_queue.put(indata.copy())
            
            # Update Visualization Data (Downsample)
            amp = np.max(np.abs(indata))
            self.waveform_data.append(amp)

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
            while self.is_running:
                sd.sleep(100)

    def _analysis_loop(self):
        """Analyze audio every 1 second using the last 3 seconds of buffer"""
        while self.is_running:
            # Consume queue into rolling buffer
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                self.audio_buffer.extend(chunk.flatten())
            
            # Analyze if we have enough data
            if len(self.audio_buffer) >= int(SAMPLE_RATE * 1.0): # Minimum 1 sec
                audio_data = np.array(self.audio_buffer)
                
                # Audio Engine
                audio_probs, audio_rms, audio_features = self.audio_engine.process_audio_data(audio_data)
                
                # Auto-Stop Logic (Silence)
                if audio_rms < 0.005: # Silence threshold
                    if self.silence_start_time is None:
                        self.silence_start_time = time.time()
                    elif time.time() - self.silence_start_time > 10.0 and self.auto_stop_silence_var.get():
                        print("[System] Auto-Stop: Silence detected for 10s.")
                        self.is_running = False
                        break
                else:
                    self.silence_start_time = None

                # Get latest Visual (from video thread)
                visual_probs = getattr(self, 'latest_visual_result', {}).get('probabilities')
                visual_landmarks = getattr(self, 'latest_visual_result', {}).get('landmarks')
                
                # Fusion (includes XAI generation)
                fusion_result = self.fusion_engine.fuse(
                    visual_probs, 
                    audio_probs,
                    visual_features=visual_landmarks if visual_landmarks else {},
                    audio_features=audio_features if audio_features else {}
                )
                
                # Store Final Result
                self.final_result = fusion_result

                # Log Data
                try:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    def get_best(probs):
                        if probs is None or len(probs) == 0: return "None", 0.0
                        if isinstance(probs, dict):
                            k = max(probs, key=probs.get)
                            return k, probs[k]
                        else:
                            # Assume numpy array or list
                            idx = np.argmax(probs)
                            if idx < len(EMOTIONS):
                                return EMOTIONS[idx], float(probs[idx])
                            return "Unknown", 0.0

                    v_e, v_c = get_best(visual_probs)
                    a_e, a_c = get_best(audio_probs)
                    f_e = fusion_result.get('emotion', 'None')
                    f_c = fusion_result.get('confidence', 0.0)
                    expl = fusion_result.get('explanation', {})
                    
                    # LOG TO DATABASE
                    if f_e != "None":
                        self.db_manager.log_emotion(self.session_id, f_e, f_c, source="fusion")

                    log_entry = f"{timestamp} | V: {v_e}({v_c:.2f}) | A: {a_e}({a_c:.2f}) | F: {f_e}({f_c:.2f}) | {expl}\n"
                    
                    with open(self.log_file, "a") as f:
                        f.write(log_entry)
                except Exception as e:
                    print(f"[Dashboard] Logging Error: {e}")
                
                # Store History for Graphs
                self.conf_history.append(fusion_result.get("confidence", 0.0))
                self.audio_conf_history.append(fusion_result.get("debug", {}).get("audio_confidence", 0.0))
                self.visual_conf_history.append(fusion_result.get("debug", {}).get("visual_confidence", 0.0))
                
                time.sleep(0.5) # Analysis rate (2 Hz)
            else:
                time.sleep(0.1)



    def _update_ui(self):
        # 1. Update Video
        if hasattr(self, 'current_image'):
            self.video_label.configure(image=self.current_image)
            self.video_label.image = self.current_image # Keep reference
        
        # 2. Update Waveform
        self.waveform_canvas.delete("all")
        w_wave = 400
        h_wave = 60
        data = list(self.waveform_data)
        if data:
            step = w_wave / len(data)
            for i, amp in enumerate(data):
                height = min(amp * 300, h_wave/2)
                x = i * step
                # Use Green for waveform
                self.waveform_canvas.create_line(x, h_wave/2 - height, x, h_wave/2 + height, fill="#2ECC71", width=1)

        # 3. Update Text/Stats
        if hasattr(self, 'final_result'):
            res = self.final_result
            emotion = res.get('emotion', '--')
            conf = res.get('confidence', 0.0)
            color_hex = res.get('color', '#95A5A6') # Default Gray
            
            expl_dict = res.get('explanation', {})
            visual_expl_text = expl_dict.get('visual', '...')
            audio_expl_text = expl_dict.get('audio', '...')
            
            modality_weights = res.get('modality_weights', {'audio': 0, 'visual': 0})
            
            # --- Update Widgets ---
            
            # Card
            self.card_frame.configure(bg=color_hex)
            emoji = EMOTION_EMOJIS.get(emotion, "")
            self.emotion_label.configure(text=f"{emoji} {emotion}", bg=color_hex)
            self.confidence_label.configure(text=f"Confidence: {int(conf*100)}%", bg=color_hex)
            
            # Confidence Bar
            self.confidence_bar['value'] = conf * 100
            
            # Modality Bars
            audio_w = modality_weights.get('audio', 0)
            visual_w = modality_weights.get('visual', 0)
            
            self.audio_contrib_label.configure(text=f"Audio: {int(audio_w*100)}%")
            self.audio_contrib_bar['value'] = audio_w * 100
            
            self.visual_contrib_label.configure(text=f"Visual: {int(visual_w*100)}%")
            self.visual_contrib_bar['value'] = visual_w * 100
            
            # Explanations
            self.visual_expl.configure(text=f"• Facial: {visual_expl_text}")
            self.audio_expl.configure(text=f"• Audio: {audio_expl_text}")

            # 4. Draw Confidence Graph
            self.graph_canvas.delete("all")
            w_graph = 400
            h_graph = 100
            
            # Draw Audio Line (Blue/Teal)
            self._draw_line_graph(self.audio_conf_history, "#3498DB", w_graph, h_graph)
            # Draw Visual Line (Green)
            self._draw_line_graph(self.visual_conf_history, "#2ECC71", w_graph, h_graph)
            # Draw Final Confidence (White, Thicker)
            self._draw_line_graph(self.conf_history, "#FFFFFF", w_graph, h_graph, width=2)
            
            # Legend
            self.graph_canvas.create_text(30, 10, text="Audio", fill="#3498DB", font=("Arial", 8))
            self.graph_canvas.create_text(80, 10, text="Visual", fill="#2ECC71", font=("Arial", 8))
            self.graph_canvas.create_text(130, 10, text="Final", fill="#FFFFFF", font=("Arial", 8))

        # Schedule next update
        if self.is_running:
            self.root.after(30, self._update_ui)
            
    def _draw_line_graph(self, data_deque, color, w, h, width=1):
        data = list(data_deque)
        if len(data) < 2: return
        
        step = w / (len(data) - 1)
        for i in range(len(data) - 1):
            y1 = h - (data[i] * h) # Invert Y
            y2 = h - (data[i+1] * h)
            x1 = i * step
            x2 = (i+1) * step
            self.graph_canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def on_close(self):
        self.is_running = False
        self.root.destroy()
        print("Dashboard Closed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MERSDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
