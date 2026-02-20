import sys
import os

# Add backend to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, "backend", "src"))
sys.path.append(os.path.join(project_dir, "backend"))

try:
    from mers.core.visual_engine import VisualEngine
    print("Import successful.")
    ve = VisualEngine()
    print("VisualEngine initialized successfully. Model found.")
except Exception as e:
    print(f"Error: {e}")
