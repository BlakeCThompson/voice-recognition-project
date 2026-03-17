# Audiobook Character Voice Classification - Usage Guide

Complete guide for using the audiobook character voice classification system.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Making Predictions](#making-predictions)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: System Requirements

**Required:**
- Python 3.8 or higher
- 8GB+ RAM
- 10GB+ disk space

**Recommended for training:**
- NVIDIA GPU with 8GB+ VRAM
- CUDA toolkit installed
- 16GB+ system RAM

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

---

## Data Preparation

### Step 1: Collect Audio Data

You need audio clips of characters speaking. Ideal sources:
- Audiobooks with distinct character voices
- Audio dramas
- Podcasts with multiple speakers

### Step 2: Organize Data

Create the following structure:

```
characters/
├── harry/
│   ├── clip_001.wav
│   ├── clip_002.wav
│   ├── ...
│   └── clip_150.wav
├── hermione/
│   ├── clip_001.wav
│   └── ...
├── ron/
│   └── ...
└── ...
```

**Guidelines:**
- **Character naming**: Use descriptive names (lowercase recommended)
- **File format**: WAV is preferred (MP3, FLAC, OGG also work)
- **Clip duration**: 4-8 seconds each
- **Number of clips**: 150-250 per character (minimum 50)
- **Audio quality**: Clear speech, minimal background noise

### Step 3: Validate and Split Data

```bash
# Run data preparation
python scripts/prepare_data.py

# This will:
# - Scan characters/ directory
# - Validate all audio files
# - Create train/val/test splits (70/15/15%)
# - Save splits to data/splits/
```

**Expected output:**
```
==================================================================
DATA PREPARATION
==================================================================

Characters directory: characters
Output directory: data/splits

Discovering character audio files...

Found 5 characters:
  harry: 200 files
  hermione: 180 files
  ron: 175 files
  dumbledore: 150 files
  hagrid: 165 files

...

DATA SPLIT STATISTICS
==================================================================

Total samples: 870
  Train: 609 (70.0%)
  Val:   130 (14.9%)
  Test:  131 (15.1%)

✓ Data preparation complete!
```

---

## Training

### Step 1: Review Configuration

Edit `configs/default.yaml` if needed:

```yaml
# Key settings
training:
  batch_size: 16          # Reduce if out of memory
  epochs: 30              # Number of training epochs
  learning_rate: 2e-5     # Learning rate

model:
  freeze_layers: 6        # Number of layers to freeze

data:
  duration: 8.0           # Audio duration in seconds
```

### Step 2: Start Training

```bash
# Train with default settings
python scripts/train.py

# Train with custom config
python scripts/train.py --config configs/custom.yaml
```

**Training output:**
```
==================================================================
CHARACTER VOICE CLASSIFICATION - TRAINING
==================================================================

Device: cuda

Loading data splits...
  Train samples: 609
  Val samples: 130
  Characters: 5

Creating model...
Model Statistics:
  Total parameters: 95,040,005
  Trainable parameters: 47,520,130
  Frozen parameters: 47,519,875

==================================================================
TRAINING START
==================================================================

Epoch 1 [Train]: 100%|████████| 38/38 [02:15<00:00]
Epoch 1 [Val]:   100%|████████| 9/9 [00:15<00:00]

Epoch 1/30
  Train Loss: 1.4523  |  Train Acc: 0.4532
  Val Loss:   1.2341  |  Val Acc:   0.5692
  ✓ New best model saved (Val Acc: 0.5692)

...

Epoch 15/30
  Train Loss: 0.2341  |  Train Acc: 0.9212
  Val Loss:   0.3456  |  Val Acc:   0.8846
  ✓ New best model saved (Val Acc: 0.8846)

==================================================================
TRAINING COMPLETE
==================================================================
Best validation accuracy: 0.8846

✓ Training complete!
```

### Step 3: Monitor Training

Training progress is automatically saved:
- **Best model**: `models/best_model.pt`
- **Checkpoints**: `models/checkpoints/epoch_XXX.pt`
- **Training curves**: `results/training_curves.png`
- **Training history**: `models/training_history.json`

### Training Tips

**Out of memory?**
```yaml
# Reduce batch size
training:
  batch_size: 8  # or even 4
```

**Training too slow?**
```yaml
# Reduce number of workers
device:
  num_workers: 2  # or 0
```

**Want faster training?**
```yaml
# Freeze more layers
model:
  freeze_layers: 9  # freeze more layers
```

---

## Evaluation

### Step 1: Evaluate on Test Set

```bash
# Evaluate best model on test set
python scripts/evaluate.py

# Evaluate on validation set
python scripts/evaluate.py --split val

# Evaluate specific checkpoint
python scripts/evaluate.py --model models/checkpoints/epoch_010.pt
```

### Step 2: Review Results

**Console output:**
```
==================================================================
EVALUATION RESULTS
==================================================================

Overall Metrics:
  Accuracy:           0.8854
  Precision (macro):  0.8723
  Recall (macro):     0.8692
  F1-Score (macro):   0.8701
  Top-3 Accuracy:     0.9769
  Top-5 Accuracy:     1.0000

Per-Character Metrics:
Character            Precision    Recall       F1-Score     Support
----------------------------------------------------------------------
dumbledore           0.9000       0.8667       0.8831       30
hagrid               0.8800       0.8800       0.8800       25
harry                0.9200       0.9200       0.9200       25
hermione             0.8400       0.9130       0.8750       23
ron                  0.8214       0.8571       0.8889       28
==================================================================
```

**Generated files:**
- `results/metrics_test.json` - Detailed metrics
- `results/confusion_matrix_test.png` - Confusion matrix
- `results/per_character_metrics_test.png` - Performance by character

### Step 3: Analyze Results

Look at confusion matrix to see which characters are confused:
- Diagonal = correct predictions (higher is better)
- Off-diagonal = confusions between characters

---

## Making Predictions

### Single File Prediction

```bash
# Predict single file (full path)
python scripts/predict.py --audio new_samples/unknown_clip.wav

# Predict single file (just filename, will look in new_samples/)
python scripts/predict.py --audio unknown_clip.wav
```

**Output:**
```
==================================================================
PREDICTION RESULT
==================================================================

Audio File: unknown_clip.wav

Predicted Character: harry
Confidence: 87.32%

All Character Probabilities:
------------------------------------------------------------------
1. harry      :  87.32%  ██████████████████████████████████████████
2. hermione   :   8.45%  ████
3. ron        :   3.12%  █
4. dumbledore :   1.11%
5. hagrid     :   0.10%
==================================================================
```

### Batch Prediction

```bash
# Predict all files in new_samples/ directory
python scripts/predict.py --batch
```

### Prediction Options

```bash
# Show only top 3 predictions
python scripts/predict.py --audio clip.wav --top_k 3

# Set confidence threshold
python scripts/predict.py --audio clip.wav --threshold 0.7

# Use custom config
python scripts/predict.py --audio clip.wav --config configs/custom.yaml
```

### Interpreting Predictions

**Confidence levels:**
- **>80%**: Very confident - highly reliable
- **60-80%**: Confident - generally reliable
- **40-60%**: Uncertain - use caution
- **<40%**: Very uncertain - unreliable

**What affects confidence:**
- Audio quality (noise, clarity)
- Speaker distinctiveness
- Training data quality/quantity
- Clip duration (longer = better, 4-8s ideal)

---

## Advanced Usage

### Custom Configuration

Create a custom config file:

```yaml
# configs/custom.yaml
data:
  duration: 10.0  # Longer clips

training:
  batch_size: 32
  epochs: 50
  learning_rate: 1e-5

model:
  freeze_layers: 9  # Freeze more layers
  dropout: 0.4
```

Use it:
```bash
python scripts/train.py --config configs/custom.yaml
```

### Resume Training

```bash
# Resume from checkpoint
python scripts/train.py --resume models/checkpoints/epoch_010.pt
```

### Data Augmentation

Customize in `configs/default.yaml`:

```yaml
augmentation:
  enabled: true
  noise:
    probability: 0.8  # Increase noise augmentation
  time_stretch:
    probability: 0.5
```

### Programmatic Usage

```python
import torch
from src.models.wav2vec2_classifier import CharacterVoiceClassifier
from src.data.preprocessing import preprocess_audio

# Load model
model = CharacterVoiceClassifier(num_characters=5)
checkpoint = torch.load('models/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess audio
waveform = preprocess_audio('new_samples/clip.wav')

# Predict
with torch.no_grad():
    logits = model(waveform.unsqueeze(0))
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax().item()

print(f"Predicted character: {pred}")
print(f"Confidence: {probs[0][pred].item():.4f}")
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory Error

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce batch size: `training.batch_size: 8`
- Use gradient accumulation: `training.accumulation_steps: 4`
- Reduce audio duration: `data.duration: 6.0`
- Use smaller model variant

#### 2. Model Not Found

**Problem:** `Error: Model not found: models/best_model.pt`

**Solution:** Train a model first:
```bash
python scripts/train.py
```

#### 3. Audio Loading Error

**Problem:** `Error: Invalid audio file`

**Solutions:**
- Convert to WAV: `ffmpeg -i input.mp3 output.wav`
- Check file is not corrupted
- Ensure minimum duration (3 seconds)

#### 4. Poor Accuracy

**Problem:** Model accuracy below 60%

**Solutions:**
- Collect more data (aim for 150+ clips per character)
- Ensure characters have distinct voices
- Check data quality (clear audio, correct labels)
- Increase training epochs
- Adjust learning rate

#### 5. Slow Training

**Problem:** Training taking too long

**Solutions:**
- Use GPU if available
- Reduce `num_workers`: `device.num_workers: 0`
- Freeze more layers: `model.freeze_layers: 9`
- Reduce dataset size for testing

### Getting Help

1. Check architecture.md for detailed documentation
2. Review configuration in configs/default.yaml
3. Verify data structure in characters/ directory
4. Check model exists in models/ directory

---

## Tips for Best Results

### Data Collection

1. **Use consistent audio source** - Same audiobook/narrator
2. **Clear audio** - Minimal background noise
3. **Distinct characters** - Choose characters with different voices
4. **Balanced dataset** - Similar number of clips per character
5. **Varied contexts** - Different emotions, speeds, dialogue types

### Training

1. **Start with defaults** - Use default config first
2. **Monitor overfitting** - Watch validation loss
3. **Early stopping** - Let it stop naturally
4. **Save checkpoints** - Keep intermediate models
5. **Experiment** - Try different hyperparameters

### Prediction

1. **Similar audio quality** - Match training data quality
2. **Appropriate length** - 4-8 seconds ideal
3. **Single speaker** - One character per clip
4. **Check confidence** - Low confidence = uncertain prediction

---

## Performance Benchmarks

### Dataset Size vs Accuracy

| Characters | Clips/Char | Total Clips | Expected Accuracy |
|------------|------------|-------------|-------------------|
| 5          | 100        | 500         | 60-70%            |
| 5          | 200        | 1,000       | 75-85%            |
| 8          | 150        | 1,200       | 70-80%            |
| 8          | 200        | 1,600       | 75-85%            |
| 10         | 200        | 2,000       | 70-80%            |
| 10         | 250        | 2,500       | 75-90%            |

### Training Time (GPU)

| Dataset Size | Epochs | Training Time |
|--------------|--------|---------------|
| 500 clips    | 30     | ~30 min       |
| 1,000 clips  | 30     | ~1 hour       |
| 1,600 clips  | 30     | ~2 hours      |
| 2,500 clips  | 30     | ~3 hours      |

*Times are approximate on NVIDIA RTX 3080*

---

## Next Steps

1. **Collect more data** for better accuracy
2. **Experiment with hyperparameters**
3. **Try different augmentation strategies**
4. **Visualize embeddings** (t-SNE)
5. **Deploy model** as web service

---

**Happy Classifying!** 🎭🎙️
