import unittest
import numpy as np
import sys
import os

# Add parent dir
sys.path.append(os.path.join(os.path.dirname(__file__), "mers"))

from mers.src.fusion_engine import FusionEngine
from mers.config import MODEL_EMOTIONS

class TestFusionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = FusionEngine()
        
    def test_audio_priority_when_conflict(self):
        """
        Test that Audio overrides Visual when Visual is weak/uncertain,
        or when Audio is strong, per user requirement.
        """
        # Scenario: 
        # Visual = Happy (but weak/uncertain, e.g. 0.35)
        # Audio = Sad (Strong, 0.8)
        # Result should be Sad
        
        visual_probs = np.zeros(len(MODEL_EMOTIONS))
        visual_probs[MODEL_EMOTIONS.index("Happy")] = 0.35
        visual_probs[MODEL_EMOTIONS.index("Neutral")] = 0.65 # Mostly neutral/uncertain
        
        audio_probs = np.zeros(len(MODEL_EMOTIONS))
        audio_probs[MODEL_EMOTIONS.index("Sad")] = 0.8
        audio_probs[MODEL_EMOTIONS.index("Neutral")] = 0.2
        
        result = self.engine.fuse(visual_probs, audio_probs)
        
        print(f"\nConflict Test 1 (Visual Weak Happy vs Audio Strong Sad): Result = {result['emotion']} (Conf: {result['confidence']:.2f})")
        self.assertEqual(result["emotion"], "Sad", "Audio should override weak visual")

    def test_visual_only(self):
        visual_probs = np.zeros(len(MODEL_EMOTIONS))
        visual_probs[MODEL_EMOTIONS.index("Happy")] = 0.9
        
        result = self.engine.fuse(visual_probs, None)
        self.assertEqual(result["emotion"], "Happy")
        self.assertEqual(result["modality_weights"]["visual"], 1.0)
        self.assertEqual(result["modality_weights"]["audio"], 0.0)

    def test_audio_only(self):
        audio_probs = np.zeros(len(MODEL_EMOTIONS))
        audio_probs[MODEL_EMOTIONS.index("Angry")] = 0.9
        
        result = self.engine.fuse(None, audio_probs)
        self.assertEqual(result["emotion"], "Angry")
        self.assertEqual(result["modality_weights"]["audio"], 1.0)
        self.assertEqual(result["modality_weights"]["visual"], 0.0)

if __name__ == "__main__":
    unittest.main()
