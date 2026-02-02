# 🚀 GPU SETUP GUIDE FOR RTX 4050 + DenseNet Training

## ✅ CURRENT SYSTEM STATUS

```
GPU:               NVIDIA GeForce RTX 4050 ✅
VRAM:              6141 MiB (6 GB) ✅
Driver Version:    561.00 ✅
CUDA Version:      12.6 ✅
Python Version:    3.13.7 ❌ NEEDS UPDATE
```

---

## 📋 STEP-BY-STEP SETUP (DO IN ORDER)

### **STEP 1: Download Python 3.10.13**

**Link**: https://www.python.org/downloads/release/python-31013/

**Or direct download**:
- Windows x86-64: https://www.python.org/ftp/python/3.10.13/python-3.10.13-amd64.exe

---

### **STEP 2: Install Python 3.10.13**

1. **Run the installer** (python-3.10.13-amd64.exe)
2. **CRITICAL**: Check ✅ "Add Python 3.10 to PATH"
3. **Install location**: Can be default (C:\Users\jo978\AppData\Local\Programs\Python\Python310\)
4. Click **Install Now**
5. Wait for completion

---

### **STEP 3: Verify Installation**

Open new PowerShell and run:

```powershell
python --version
```

Expected output:
```
Python 3.10.13
```

If still shows 3.13, restart PowerShell or check PATH.

---

### **STEP 4: Create Fresh GPU Virtual Environment**

Navigate to project:

```powershell
cd C:\Users\jo978\OneDrive\Documents\GitHub\projectworkII
```

Remove old venv:

```powershell
Remove-Item .venv -Recurse -Force
```

Create new venv with Python 3.10:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

You should see:
```
(.venv) PS C:\Users\...>
```

---

### **STEP 5: Upgrade pip**

```powershell
python -m pip install --upgrade pip
```

---

### **STEP 6: Install GPU PyTorch (CUDA 11.8)**

Run **exactly this** (it's one long command):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

⏱️ Takes 2-3 minutes  
📦 ~2.5 GB download

Wait for completion. You should see:
```
Successfully installed torch-2.x.x ...
```

---

### **STEP 7: Install other dependencies**

```powershell
pip install numpy tqdm joblib pywavelets pillow
```

---

### **STEP 8: Verify GPU Detection**

Run Python:

```powershell
python
```

Then paste this code:

```python
import torch
print("=" * 60)
print("PyTorch GPU Verification")
print("=" * 60)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {props.total_memory / 1024**3:.1f} GB")
print("=" * 60)
```

**Expected output**:
```
============================================================
PyTorch GPU Verification
============================================================
CUDA Available: True
CUDA Version: 11.8
GPU Count: 1
GPU Name: NVIDIA GeForce RTX 4050
GPU Memory: 6.0 GB
============================================================
```

Exit Python:
```python
exit()
```

---

## ✅ Verification Checklist

- [ ] Python 3.10.13 installed
- [ ] `.venv` created and activated (shows `(.venv)` in terminal)
- [ ] PyTorch CUDA 11.8 installed
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] GPU name shows as RTX 4050
- [ ] GPU memory shows ~6.0 GB

---

## 🔥 Once ALL checks pass:

Say: **"GPU READY - PROCEED WITH CODE OPTIMIZATION"**

And I will:
1. ✅ Optimize densenet_classifier.py for RTX 4050
2. ✅ Set batch size to 32
3. ✅ Enable pin_memory for DataLoader
4. ✅ Freeze early DenseNet layers
5. ✅ Run training on GPU
6. ✅ Collect accuracy metrics

---

## 📝 Troubleshooting

**Issue**: `python --version` still shows 3.13
- **Fix**: Close all terminals and open new one. Check PATH in System Environment Variables.

**Issue**: PyTorch says CUDA not available
- **Fix**: Restart computer after Python installation. Run `pip install --force-reinstall torch` in venv.

**Issue**: "Add to PATH" not checked during install
- **Fix**: Reinstall Python 3.10.13, make sure to check that checkbox.

---

## 📞 Report back:

Once verification complete, say:
> "✅ GPU SETUP COMPLETE - [CUDA True/False] - [GPU Name] - [Memory in GB]"

Then we proceed! 🚀
