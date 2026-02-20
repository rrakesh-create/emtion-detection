import cv2
import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class UIRenderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Precompute colors
        self.colors = {k: v for k, v in EMOTION_COLORS.items()}
        self.timeline_history = [] 
        self.max_timeline_len = 200 

    def draw_stability_bar(self, canvas, stability, x, y, w):
        """
        Draws the face embedding stability bar.
        """
        h = 10
        cv2.putText(canvas, "Face Stability", (int(x), int(y - 5)), self.font, 0.5, (200, 200, 200), 1)
        
        # Background
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (50, 50, 50), -1)
        
        # Foreground
        bar_len = int(stability * w)
        # Color based on value
        color = (0, 0, 255) # Red (Low)
        if stability > 0.4: color = (0, 165, 255) # Orange
        if stability > 0.7: color = (0, 255, 0) # Green
        
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + bar_len), int(y + h)), color, -1)
        cv2.putText(canvas, f"{stability*100:.0f}%", (int(x + w + 5), int(y + h)), self.font, 0.4, color, 1)

    def draw_loading_screen(self, steps):
        """
        steps: list of (description, status)
        status: "pending", "running", "done", "error"
        """
        canvas = np.zeros((UI_HEIGHT, UI_WIDTH, 3), dtype=np.uint8)
        canvas[:] = BG_COLOR
        
        # Title
        cv2.putText(canvas, f"{APP_NAME} v{VERSION}", (50, 80), self.font, 1.5, ACCENT_COLOR, 3)
        cv2.putText(canvas, "System Initialization", (50, 130), self.font, 0.8, (200, 200, 200), 1)
        
        start_y = 200
        for i, (desc, status) in enumerate(steps):
            y = start_y + i * 50
            
            color = (150, 150, 150) # Pending
            icon = "[ ]"
            if status == "running":
                color = (0, 255, 255) # Yellow
                icon = "[...]"
            elif status == "done":
                color = (0, 255, 0) # Green
                icon = "[OK]"
            elif status == "error":
                color = (0, 0, 255) # Red
                icon = "[X]"
            
            cv2.putText(canvas, f"{icon} {desc}", (50, y), self.font, 0.7, color, 2)
            
        cv2.imshow(WINDOW_NAME, canvas)
        cv2.waitKey(1)

    def draw_session_summary(self, total_time, dominant_emotion, conflict_count, emotion_counts):
        """
        Display summary stats.
        """
        canvas = np.zeros((UI_HEIGHT, UI_WIDTH, 3), dtype=np.uint8)
        canvas[:] = BG_COLOR
        
        cv2.putText(canvas, "Session Summary", (50, 80), self.font, 1.5, ACCENT_COLOR, 3)
        
        # Stats
        cv2.putText(canvas, f"Total Duration: {total_time:.2f} sec", (50, 150), self.font, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Dominant Emotion: {dominant_emotion}", (50, 200), self.font, 0.8, self.colors.get(dominant_emotion, (255, 255, 255)), 2)
        cv2.putText(canvas, f"Conflicts Detected: {conflict_count}", (50, 250), self.font, 0.8, (0, 0, 255) if conflict_count > 0 else (0, 255, 0), 2)
        
        # Breakdown
        cv2.putText(canvas, "Emotion Breakdown:", (50, 320), self.font, 0.7, (200, 200, 200), 1)
        
        y = 360
        max_count = max(emotion_counts.values()) if emotion_counts else 1
        
        for i, emo in enumerate(EMOTIONS):
            count = emotion_counts.get(emo, 0)
            bar_len = int((count / max_count) * 400)
            color = self.colors[emo]
            
            cv2.putText(canvas, f"{emo}: {count}", (50, y), self.font, 0.6, (200, 200, 200), 1)
            cv2.rectangle(canvas, (200, y - 15), (200 + bar_len, y + 5), color, -1)
            y += 35
            
        cv2.putText(canvas, "Press ESC to Close", (50, UI_HEIGHT - 50), self.font, 0.8, (0, 255, 0), 2)
        
        cv2.imshow(WINDOW_NAME, canvas)
        # Block until ESC
        while True:
            if cv2.waitKey(100) & 0xFF == 27:
                break
        
    def draw_ui(self, frame, audio_waveform, visual_probs, audio_probs, fused_probs, 
                face_box, emotion_label, stability, conflict, fps, log_entries):
        """
        Main rendering function.
        frame: Webcam frame (BGR)
        audio_waveform: numpy array of audio samples
        probs: dictionaries or arrays of probabilities
        """
        
        # 1. Create Canvas
        canvas = np.zeros((UI_HEIGHT, UI_WIDTH, 3), dtype=np.uint8)
        canvas[:] = BG_COLOR
        
        # 2. Left Panel: Camera Feed
        # Resize frame to fit left half 
        # Target: (UI_WIDTH // 2, UI_HEIGHT * 0.7) approx?
        # Let's say Left Panel is 600px wide.
        
        left_w = UI_WIDTH // 2
        right_w = UI_WIDTH - left_w
        
        if frame is not None:
            # Resize frame to fit width of left panel while maintaining aspect ratio
            h, w = frame.shape[:2]
            scale = left_w / w
            new_h = int(h * scale)
            resized_frame = cv2.resize(frame, (left_w, new_h))
            
            # Center vertically in left panel
            y_offset = (UI_HEIGHT - new_h) // 2
            canvas[y_offset:y_offset+new_h, 0:left_w] = resized_frame
            
            # Draw Face Box
            if face_box is not None:
                x1, y1, x2, y2 = face_box
                # Scale coordinates
                x1 = int(x1 * scale)
                y1 = int(y1 * scale) + y_offset
                x2 = int(x2 * scale)
                y2 = int(y2 * scale) + y_offset
                
                color = self.colors.get(emotion_label, (255, 255, 255))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, f"{emotion_label} ({stability*100:.0f}%)", (x1, y1 - 10), 
                            self.font, 0.7, color, 2)

        # 3. Right Panel: Analytics
        
        # A. Audio Waveform (Top)
        viz_h = 150
        cv2.rectangle(canvas, (left_w + 10, 10), (UI_WIDTH - 10, 10 + viz_h), (50, 50, 50), -1)
        if audio_waveform is not None:
            # Normalize for visualization
            # wav is typically -1 to 1?
            # draw line
            
            # Downsample for speed
            step = max(1, len(audio_waveform) // (right_w - 20))
            pts = []
            center_y = 10 + viz_h // 2
            
            for i, val in enumerate(audio_waveform[::step]):
                x = left_w + 10 + i
                if x >= UI_WIDTH - 10: break
                y = int(center_y + val * (viz_h / 2))
                y = max(10, min(10 + viz_h, y))
                pts.append((x, y))
            
            if len(pts) > 1:
                cv2.polylines(canvas, [np.array(pts)], False, (0, 255, 0), 1)

        cv2.putText(canvas, "Audio Input", (left_w + 20, 30), self.font, 0.5, (200, 200, 200), 1)

        # B. Probability Bars (Middle)
        # Visual vs Audio vs Fused
        
        bar_start_y = 180
        bar_h = 20
        gap = 35
        
        # Columns
        col_w = (right_w - 40) // 3
        
        headers = ["Visual", "Audio", "Fused"]
        for i, h in enumerate(headers):
            cv2.putText(canvas, h, (left_w + 20 + i*col_w, bar_start_y), self.font, 0.6, (255, 255, 255), 2)
            
        bar_start_y += 30
        
        for i, emo in enumerate(EMOTIONS):
            y = bar_start_y + i * gap
            color = self.colors[emo]
            
            # Label
            cv2.putText(canvas, emo[:3], (left_w + 5, y + 15), self.font, 0.5, (150, 150, 150), 1)
            
            # Bars
            max_w = col_w - 20
            
            # Visual
            if visual_probs is not None:
                v_len = int(visual_probs[i] * max_w)
                cv2.rectangle(canvas, (left_w + 40, y), (left_w + 40 + v_len, y + bar_h), color, -1)
                
            # Audio
            if audio_probs is not None:
                a_len = int(audio_probs[i] * max_w)
                cv2.rectangle(canvas, (left_w + 40 + col_w, y), (left_w + 40 + col_w + a_len, y + bar_h), color, -1)
                
            # Fused
            if fused_probs is not None:
                f_len = int(fused_probs[i] * max_w)
                cv2.rectangle(canvas, (left_w + 40 + 2*col_w, y), (left_w + 40 + 2*col_w + f_len, y + bar_h), color, -1)
                # Highlight winner
                if np.argmax(fused_probs) == i:
                     cv2.rectangle(canvas, (left_w + 40 + 2*col_w, y), (left_w + 40 + 2*col_w + f_len, y + bar_h), (255, 255, 255), 1)

        # Stability Bar (Below Probability Bars)
        self.draw_stability_bar(canvas, stability, left_w + 20, bar_start_y + 7.5 * gap, right_w - 40)

        # C. Conflict Warning
        if conflict:
            # Flashing Border
            if int(time.time() * 5) % 2 == 0: # Flash ~5 times a second
                cv2.rectangle(canvas, (0, 0), (UI_WIDTH-1, UI_HEIGHT-1), (0, 0, 255), 10)
            
            cv2.putText(canvas, "CONFLICT DETECTED!", (left_w + 20, bar_start_y + 8 * gap), self.font, 1.0, (0, 0, 255), 3)
            
            # Detailed Conflict Info
            cv2.putText(canvas, "Visual vs Audio Mismatch", (left_w + 20, bar_start_y + 8 * gap + 30), self.font, 0.6, (150, 150, 255), 1)

        # D. Logs (Bottom)
        log_y = bar_start_y + 9 * gap
        cv2.line(canvas, (left_w, log_y), (UI_WIDTH, log_y), (100, 100, 100), 1)
        
        for i, log in enumerate(reversed(log_entries[-5:])): # Show last 5
            cv2.putText(canvas, log, (left_w + 10, log_y + 20 + i*20), self.font, 0.4, (200, 200, 200), 1)
        # E. Timeline Strip (Bottom)
        timeline_y = UI_HEIGHT - 40
        self.update_timeline(emotion_label, conflict)
        
        # Draw timeline
        # Each block is a small rectangle
        block_w = UI_WIDTH / self.max_timeline_len
        
        for i, (emo, color) in enumerate(self.timeline_history):
            x = int(i * block_w)
            # If conflict, maybe add a marker?
            cv2.rectangle(canvas, (x, timeline_y), (int(x + block_w), UI_HEIGHT), color, -1)
            
        cv2.putText(canvas, "Timeline", (10, timeline_y - 5), self.font, 0.4, (150, 150, 150), 1)

        # FPS
        cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 20), self.font, 0.6, (0, 255, 0), 2)

        return canvas

    def update_timeline(self, emotion, conflict):
        if emotion is None: return
        
        color = self.colors.get(emotion, (50, 50, 50))
        # If conflict, maybe flash or different color? 
        # For now just use emotion color.
        
        self.timeline_history.append((emotion, color))
        if len(self.timeline_history) > self.max_timeline_len:
            self.timeline_history.pop(0)
