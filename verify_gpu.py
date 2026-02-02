import torch

print("=" * 60)
print("PyTorch GPU Verification")
print("=" * 60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    print(f"GPU Memory: {vram_gb:.1f} GB")
print("=" * 60)
