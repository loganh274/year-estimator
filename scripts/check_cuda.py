import torch
import sys

print("=== CUDA Diagnostics ===")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("❌ CUDA is NOT available. PyTorch is running on CPU.")
    print("\nPossible causes:")
    print("1. PyTorch installed without CUDA support (CPU-only version).")
    print("2. NVIDIA Drivers are missing or outdated.")
    print("3. No NVIDIA GPU detected.")
    
    print("\nTo install PyTorch with CUDA support, run:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
