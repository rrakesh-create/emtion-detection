
import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import queue
import threading
import time

# --- Configuration ---
WINDOW_NAME = "SynEmotion V0 - Simple Demo"
AUDIO_RATE = 16000
AUDIO_BLOCK_SIZE = 1024
EMOTIONS = ["Angry", "Happy", "Sad", "Neutral", "Fear", "Surprise", "Disgust"]

# --- Global State ---
audio_queue = queue.Queue()
current_energy = 0.0
stop_threads = False

# --- Audio Callback ---
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    # Convert to mono and calculate RMS energy
    audio_data = indata[:, 0]
    rms = np.sqrt(np.mean(audio_data**2))
    audio_queue.put(rms)

# --- Audio Processing Loop ---
def audio_processing():
    global current_energy
    while not stop_threads:
        try:
            # Get latest energy
            rms = audio_queue.get(timeout=0.1)
            # Simple smoothing
            current_energy = 0.7 * current_energy + 0.3 * rms
        except queue.Empty:
            pass

# --- Helper: Get Dummy Emotion based on Audio ---
def get_dummy_audio_emotion(energy):
    # Very simple heuristic for demo purposes
    if energy < 0.01:
        return "Neutral" # Silence
    elif energy < 0.05:
        return "Sad"     # Quiet speech
    elif energy < 0.2:
        return "Happy"   # Normal speech
    else:
        return "Angry"   # Loud speech / shouting

# --- Main Demo ---
def main():
    global stop_threads
    
    # 1. Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 2. Start Audio Stream
    try:
        stream = sd.InputStream(
            channels=1,
            samplerate=AUDIO_RATE,
            callback=audio_callback,
            blocksize=AUDIO_BLOCK_SIZE
        )
        stream.start()
        print("Audio stream started.")
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        return

    # 3. Start Audio Thread
    audio_thread = threading.Thread(target=audio_processing)
    audio_thread.start()

    # 4. Start Video Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Main Loop. Press 'ESC' to exit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        h, w, c = image.shape
        
        # --- Visualization: Audio Bar ---
        # Draw a bar on the right side representing audio energy
        bar_height = int(current_energy * h * 5) # Scale energy
        bar_height = min(bar_height, h) # Clip
        cv2.rectangle(image, (w - 50, h - bar_height), (w, h), (0, 255, 0), -1)
        cv2.putText(image, "Audio", (w - 60, h - bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- Visualization: Dummy Emotion ---
        audio_emotion = get_dummy_audio_emotion(current_energy)
        cv2.putText(image, f"Audio Impulse: {audio_emotion}", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, "Visual: [Detecting Lines...]", (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw tesselation
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # Draw contours
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                
                # Draw irises
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        cv2.imshow(WINDOW_NAME, image)

        if cv2.waitKey(5) & 0xFF == 27:
            stop_threads = True
            break

    cap.release()
    stream.stop()
    stream.close()
    cv2.destroyAllWindows()
    stop_threads = True
    audio_thread.join()

if __name__ == "__main__":
    main()
