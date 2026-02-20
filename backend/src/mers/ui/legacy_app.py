import cv2
import threading
import time
import numpy as np
import sys
import os

# Add MERS to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from src.visual_engine import VisualEngine
from src.audio_engine import AudioEngine
from src.fusion_engine import FusionEngine
from src.ui_renderer import UIRenderer
from src.logger import EmotionLogger

def main():
    # 0. Init UI Renderer immediately for loading screen
    ui_renderer = UIRenderer()
    
    # Steps for loading screen
    init_steps = [
        ("Initializing UI...", "running"),
        ("Checking CUDA/Device...", "pending"),
        ("Loading Audio Engine...", "pending"),
        ("Loading Visual Engine...", "pending"),
        ("Starting Fusion Engine...", "pending"),
        ("Connecting Camera...", "pending")
    ]
    
    def update_step(idx, status):
        init_steps[idx] = (init_steps[idx][0], status)
        ui_renderer.draw_loading_screen(init_steps)
        time.sleep(0.1) # Debounce for visual effect

    update_step(0, "done")
    
    # 1. Device Check
    update_step(1, "running")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}")
    update_step(1, "done")
    
    # 2. Engines
    try:
        update_step(2, "running")
        audio_engine = AudioEngine()
        update_step(2, "done")
        
        update_step(3, "running")
        visual_engine = VisualEngine()
        update_step(3, "done")
        
        update_step(4, "running")
        fusion_engine = FusionEngine()
        logger = EmotionLogger()
        update_step(4, "done")
    except Exception as e:
        print(f"Initialization Failed: {e}")
        # Mark current as error
        # In real app, we should show error on UI
        return

    # 3. Audio Thread
    state = {
        "visual_probs": None,
        "face_box": None,
        "embedding_stability": 0.0,
        "audio_probs": None,
        "audio_rms": 0.0,
        "audio_waveform": None,
        "running": True
    }
    state_lock = threading.Lock()

    def audio_worker():
        audio_engine.start_stream()
        while state["running"]:
            probs, rms, waveform = audio_engine.process_queue()
            if probs is not None or waveform is not None:
                with state_lock:
                    if probs is not None:
                        state["audio_probs"] = probs
                    state["audio_rms"] = rms
                    state["audio_waveform"] = waveform
            time.sleep(0.05)
        audio_engine.stop_stream()

    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()

    # 4. Camera
    update_step(5, "running")
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        update_step(5, "error")
        time.sleep(2)
        return
    update_step(5, "done")

    # Session Stats
    session_start = time.time()
    emotion_counts = {e: 0 for e in EMOTIONS}
    total_conflicts = 0
    frames_processed = 0

    last_visual_time = 0
    start_time = time.time()
    frame_count = 0
    fps = 0.0

    print("System active. Press 'ESC' to exit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break

            current_time = time.time()
            frame_count += 1
            if current_time - start_time > 1.0:
                fps = frame_count / (current_time - start_time)
                frame_count = 0
                start_time = current_time

            # Visual Inference
            visual_probs = state["visual_probs"]
            face_box = state["face_box"]
            stability = state["embedding_stability"]
            
            if current_time - last_visual_time > VISUAL_INFERENCE_INTERVAL:
                v_probs, f_box, v_stab = visual_engine.process_frame(frame)
                with state_lock:
                    if v_probs is not None:
                        state["visual_probs"] = v_probs
                        state["face_box"] = f_box
                        state["embedding_stability"] = v_stab
                    else:
                        state["visual_probs"] = None
                        state["face_box"] = None
                        state["embedding_stability"] = 0.0
                last_visual_time = current_time

            # Retrieve State
            with state_lock:
                v_probs_curr = state["visual_probs"]
                a_probs_curr = state["audio_probs"]
                stab_curr = state["embedding_stability"]
                box_curr = state["face_box"]
                wav_curr = state["audio_waveform"]
            
            # Fuse
            fused, conflict, label, source = fusion_engine.fuse(v_probs_curr, a_probs_curr, stab_curr)
            
            # Stats Logging
            if label != "Waiting...":
                frames_processed += 1
                logger.log_csv(label, np.max(fused) if fused is not None else 0, conflict, source)
                if frames_processed % 5 == 0: 
                    emotion_counts[label] = emotion_counts.get(label, 0) + 1
            
            if conflict:
                total_conflicts += 1 # This might overcount per frame. Maybe debounce?
                logger.log(label, np.max(fused) if fused is not None else 0, conflict, source)
                
            # Render
            ui_frame = ui_renderer.draw_ui(
                frame, wav_curr, v_probs_curr, a_probs_curr, fused, 
                box_curr, label, stab_curr, conflict, fps, logger.get_logs()
            )

            cv2.imshow(WINDOW_NAME, ui_frame)

            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...")
        state["running"] = False
        audio_thread.join()
        cap.release()
        
        # Show Session Summary
        session_duration = time.time() - session_start
        # Determine dominant emotion
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if any(emotion_counts.values()) else "None"
        
        # We need to render this
        ui_renderer.draw_session_summary(session_duration, dominant_emotion, total_conflicts // 10, emotion_counts) # Divide conflicts by approx fps factor if counted every frame? Let's just show raw count or debounce logic needed. 
        # Actually simplest is just show 'frames with conflict'? 
        
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
