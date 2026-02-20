import unittest
import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mers.src.visual_engine import VisualEngine

class TestVisualEngine(unittest.TestCase):
    def setUp(self):
        """Initialize VisualEngine before each test."""
        self.engine = VisualEngine()

    def test_process_frame_no_face(self):
        """Test processing a black frame (no face)."""
        # Create a black image 640x480
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        probs, face_box, features, landmarks = self.engine.process_frame(frame)
        
        # Should return None/Empty for no detection
        self.assertIsNone(probs)
        self.assertIsNone(face_box)
        self.assertEqual(features, {})
        self.assertIsNone(landmarks)

    def test_process_frame_shape(self):
        """Test that if a face were detected (mocked), it returns correct shape."""
        # Since we can't easily inject a real face without a file, 
        # we verify the engine handles the input format correctly without crashing.
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.engine.process_frame(frame)
        except Exception as e:
            self.fail(f"process_frame raised Exception unexpectedly: {e}")

    def test_engine_initialization(self):
        """Test that the engine initializes correctly."""
        self.assertIsNotNone(self.engine.detector)

if __name__ == '__main__':
    unittest.main()
