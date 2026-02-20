import unittest
import sys
import os
import numpy as np

# Add project root to path. 
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "mers"))

from mers.src.fusion_engine import FusionEngine
from mers.config import EMOTIONS, MODEL_EMOTIONS

class TestFusionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = FusionEngine()
        # Model Emotions: ["Angry", "Happy", "Sad", "Neutral", "Fear", "Surprise", "Disgust"]

    def test_fusion_logic_balanced(self):
        # Happy at index 1
        # Visual: Happy (0.8)
        v_probs = np.zeros(7)
        v_probs[1] = 0.8
        
        # Audio: Happy (0.8)
        a_probs = np.zeros(7)
        a_probs[1] = 0.8
        
        # Expect Balanced Weights (0.5, 0.5) because a_conf (0.8) is not > v_conf (0.8)
        # Actually logic says: if a_conf > v_conf: Audio Dominates. Else: Balanced.
        # Here 0.8 is not > 0.8, so Balanced.
        
        result = self.engine.fuse(v_probs, a_probs, {}, {})
        
        self.assertEqual(result["emotion"], "Happy")
        self.assertEqual(result["modality_weights"]["visual"], 0.5)
        self.assertEqual(result["modality_weights"]["audio"], 0.5)

    def test_audio_priority(self):
        # Visual: Angry (Index 0) - Weak (0.4)
        v_probs = np.zeros(7)
        v_probs[0] = 0.4
        
        # Audio: Happy (Index 1) - Strong (0.9)
        a_probs = np.zeros(7)
        a_probs[1] = 0.9
        
        # Logic: if v_conf < 0.5 -> Audio Dominates (0.1, 0.9)
        
        result = self.engine.fuse(v_probs, a_probs, {}, {})
        
        self.assertEqual(result["emotion"], "Happy") # Audio wins
        self.assertEqual(result["modality_weights"]["audio"], 0.9)
        self.assertEqual(result["modality_weights"]["visual"], 0.1)

    def test_visual_only(self):
        # Visual: Sad (Index 2) - Strong
        v_probs = np.zeros(7)
        v_probs[2] = 0.9
        
        result = self.engine.fuse(v_probs, None, {}, {})
        
        self.assertEqual(result["emotion"], "Sad")
        self.assertEqual(result["modality_weights"]["visual"], 1.0)
        self.assertEqual(result["modality_weights"]["audio"], 0.0)

    def test_audio_only(self):
        # Audio: Neutral (Index 3) - Strong
        a_probs = np.zeros(7)
        a_probs[3] = 0.9
        
        result = self.engine.fuse(None, a_probs, {}, {})
        
        self.assertEqual(result["emotion"], "Neutral")
        self.assertEqual(result["modality_weights"]["audio"], 1.0)
        self.assertEqual(result["modality_weights"]["visual"], 0.0)

    def test_mapping_surprise_to_focused(self):
        # Model predicts Surprise (Index 5)
        v_probs = np.zeros(7)
        v_probs[5] = 0.9
        
        result = self.engine.fuse(v_probs, None, {}, {})
        
        self.assertEqual(result["emotion"], "Focused")

    def test_mapping_disgust_to_stressed(self):
        # Model predicts Disgust (Index 6)
        v_probs = np.zeros(7)
        v_probs[6] = 0.9
        
        result = self.engine.fuse(v_probs, None, {}, {})
        
        self.assertEqual(result["emotion"], "Stressed")

if __name__ == '__main__':
    unittest.main()
