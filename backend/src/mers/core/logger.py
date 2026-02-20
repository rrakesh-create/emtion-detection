import logging
import os
import csv
import time
from config import BASE_DIR

class EmotionLogger:
    def __init__(self):
        # Text Logger
        self.log_file = os.path.join(BASE_DIR, "emotion_log.txt")
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger()
        self.log_cache = [] 
        self.MAX_CACHE = 10
        
        # CSV Logger
        self.metrics_dir = os.path.join(BASE_DIR, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.csv_file = os.path.join(self.metrics_dir, f"session_{int(time.time())}.csv")
        
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Emotion", "Confidence", "Source", "Conflict"])

    def log(self, emotion, confidence, conflict=False, source="Multimodal"):
        # Text Log
        conflict_tag = "[Conflict]" if conflict else ""
        msg = f"Emotion: {emotion} | Confidence: {confidence:.2f} | Src: {source} {conflict_tag}"
        self.logger.info(msg)
        
        # UI Cache
        self.log_cache.append(msg)
        if len(self.log_cache) > self.MAX_CACHE:
            self.log_cache.pop(0)

    def log_csv(self, emotion, confidence, conflict, source):
        # Separate method for high-frequency CSV logging
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), emotion, confidence, source, conflict])

    def get_logs(self):
        return self.log_cache
