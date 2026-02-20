import sys
import os

import tkinter as tk

# Add backend/src and backend/config to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, "backend", "src"))
sys.path.append(os.path.join(project_dir, "backend"))

def start_ui():
    from mers.ui.dashboard import MERSDashboard
    
    print("[MERS] Starting Application...")
    try:
        root = tk.Tk()
        app = MERSDashboard(root)
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        root.mainloop()
    except Exception as e:
        print(f"[MERS] Error: {e}")
        input("Press Enter to exit...")



def main():
    print("[MERS] Initializing...")
    mode = "ui"

    if mode is None:
        print("\n==========================================")
        print("   MERS Multimodal Emotion Recognition    ")
        print("==========================================")
        print("Starting PC Dashboard (Webcam Input)...")
        mode = "ui"

    if mode == "ui":
        start_ui()
    # Server and Viewer modes removed as per cleanup request

if __name__ == "__main__":
    main()
