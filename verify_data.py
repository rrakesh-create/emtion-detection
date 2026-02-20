import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mers.train.train_audio import UnifiedAudioDataset
from mers.config import EMOTIONS

def verify_datasets():
    # mers/train/verify_data.py -> mers/train -> mers -> datasets
    # But wait, verify_data.py is in mers/ directly? No, I'll place it in project root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base_dir, "datasets")
    
    print(f"Checking datasets in {data_root}...")
    dataset = UnifiedAudioDataset(data_root, EMOTIONS, augment=True)
    
    print(f"Total samples: {len(dataset)}")
    
    # Check distribution
    counts = {e: 0 for e in EMOTIONS}
    # Reverse map
    idx_to_class = {i: e for i, e in enumerate(EMOTIONS)}
    
    for _, label_idx in dataset:
        # Note: dataset returns tensors, we just need labels here. 
        # But calling __getitem__ loads audio which is slow.
        # We can access dataset.labels directly if it's populated.
        break
        
    for label in dataset.labels:
        class_name = idx_to_class[label]
        counts[class_name] += 1
        
    print("Emotion distribution:")
    for e, c in counts.items():
        print(f"  {e}: {c}")

if __name__ == "__main__":
    verify_datasets()
