# Installation Guide

## Prerequisites

### System Requirements

**Minimum:**
- Python 3.8 or higher
- 8 GB RAM
- 10 GB disk space

**Recommended:**
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.7+ and cuDNN

### Check Python Version

```bash
python --version
# Should show Python 3.8 or higher
```

---

## Installation Steps

### Option 1: Quick Install (Recommended)

```bash
# 1. Navigate to project directory
cd audiobook-character-classifier

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Option 2: Install with Package Manager

```bash
# With pip
pip install -e .

# This installs the package and creates command-line tools:
# - audiobook-prepare
# - audiobook-train
# - audiobook-evaluate
# - audiobook-predict
```

---

## GPU Setup (Optional but Recommended)

### Check CUDA Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### Install GPU-Enabled PyTorch

If CUDA is not detected, reinstall PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Troubleshooting Installation

### Issue: "No module named torch"

**Solution:**
```bash
pip install torch torchaudio
```

### Issue: "No module named transformers"

**Solution:**
```bash
pip install transformers
```

### Issue: Out of Memory During Installation

**Solution:**
```bash
# Install with no cache
pip install --no-cache-dir -r requirements.txt
```

### Issue: Permission Denied

**Solution:**
```bash
# Install for current user only
pip install --user -r requirements.txt
```

### Issue: SSL Certificate Error

**Solution:**
```bash
# Install with trusted host
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

---

## Verify Installation

Run this verification script:

```bash
python << 'VERIFY'
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except:
    print("✗ PyTorch not installed")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except:
    print("✗ Transformers not installed")

try:
    import torchaudio
    print(f"✓ Torchaudio: {torchaudio.__version__}")
except:
    print("✗ Torchaudio not installed")

try:
    import librosa
    print(f"✓ Librosa: {librosa.__version__}")
except:
    print("✗ Librosa not installed")

try:
    import sklearn
    print(f"✓ Scikit-learn: {sklearn.__version__}")
except:
    print("✗ Scikit-learn not installed")

print("\n✓ Installation verified!")
VERIFY
```

Expected output:
```
Python: 3.10.x
✓ PyTorch: 2.1.0
  CUDA available: True
✓ Transformers: 4.35.0
✓ Torchaudio: 2.1.0
✓ Librosa: 0.10.1
✓ Scikit-learn: 1.3.2

✓ Installation verified!
```

---

## Alternative: Docker Installation (Advanced)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
```

Build and run:
```bash
docker build -t audiobook-classifier .
docker run -it -v $(pwd)/characters:/app/characters audiobook-classifier
```

---

## Next Steps

After successful installation:

1. **Verify project structure** exists:
   ```bash
   ls -la
   # Should see: characters/, configs/, scripts/, src/
   ```

2. **Read the documentation**:
   - Quick start: `QUICKSTART.md`
   - Full guide: `USAGE_GUIDE.md`
   - Architecture: `architecture.md`

3. **Prepare your data**:
   - Organize audio files in `characters/` directory
   - Run: `python scripts/prepare_data.py`

---

## Updating Dependencies

To update to latest versions:

```bash
pip install --upgrade -r requirements.txt
```

To check for outdated packages:

```bash
pip list --outdated
```

---

## Uninstallation

To completely remove:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv

# Remove installed package (if installed with pip install -e .)
pip uninstall audiobook-character-classifier
```

---

**Installation Complete!** You're ready to build your character voice classifier. 🎉
