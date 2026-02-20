
import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA is NOT available. Using CPU.")
except ImportError:
    print("PyTorch is NOT installed.")

try:
    import torchvision
    print(f"Torchvision Version: {torchvision.__version__}")
except ImportError:
    print("Torchvision is NOT installed.")
