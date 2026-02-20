# MERS Face Visualization System

## Overview
The Face Visualization System in MERS provides real-time visual feedback on the detected user's face, including emotion-coded borders and facial mesh mapping. This system is designed to run locally on the client (Dashboard) to ensure privacy and low latency.

## Features

### 1. Dynamic Emotion Borders
- **Functionality**: Draws a bounding box around the detected face.
- **Color Coding**: The border color changes dynamically based on the detected emotion.
  - Happy: Green (`#2ECC71`)
  - Sad: Blue (`#3498DB`)
  - Angry: Red (`#E74C3C`)
  - Fear: Purple (`#9B59B6`)
  - Neutral: Gray (`#95A5A6`)
  - Focused: Teal (`#1ABC9C`)
  - Stressed: Orange (`#E67E22`)
- **Confidence Display**: Shows the emotion label and confidence percentage above the box.

### 2. Face Mesh Mapping
- **Functionality**: Overlays facial landmarks (468 points) on the video feed.
- **Usage**: Useful for verifying that the system is correctly tracking facial features (eyes, lips, eyebrows).
- **Toggle**: Can be enabled/disabled via the "Show Face Mesh" checkbox in the dashboard.

## Configuration
The visualization settings can be toggled in real-time via the Dashboard UI:
- **Show Face Box**: Toggles the emotion-colored bounding box.
- **Show Face Mesh**: Toggles the raw mesh points.

## Architecture & Privacy
- **Local Processing**: All face detection and rendering happen locally using MediaPipe and OpenCV. No video data is sent to the cloud.
- **Performance**: Optimized for 30+ FPS on standard laptop webcams.
- **Integration**: The `VisualEngine` class returns raw landmarks which the `MERSDashboard` uses for rendering.

## API Reference (VisualEngine)

### `process_frame(frame)`
Processes a single video frame.

**Input**:
- `frame`: numpy array (BGR image from OpenCV)

**Output**:
- `emotion_probs`: Array of probabilities for each emotion.
- `face_box`: Tuple `(x1, y1, x2, y2)` of the bounding box.
- `features`: Dictionary of geometric features (e.g., eye_opening, mouth_ratio).
- `landmarks`: List of normalized landmarks (for drawing mesh).

## Integration Example

```python
from mers.src.visual_engine import VisualEngine
import cv2

engine = VisualEngine()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    probs, box, features, landmarks = engine.process_frame(frame)
    
    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    cv2.imshow("MERS Face Viz", frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()
```
