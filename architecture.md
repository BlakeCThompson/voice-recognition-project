# Audiobook Character Voice Classification - System Architecture

## Project Overview

A deep learning system that classifies fictional characters from audiobook narration based on voice characteristics. The system uses transfer learning with pretrained speech models to achieve high accuracy with limited training data.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                              │
│                     (CLI Prediction Script)                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 │ Audio File Path
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐   │
│  │ Audio Loader   │→ │ Preprocessor   │→ │ Feature Extractor │   │
│  └────────────────┘  └────────────────┘  └────────────────────┘   │
│                                                    │                 │
│                                                    ▼                 │
│                             ┌─────────────────────────────────┐     │
│                             │   Pretrained Model (wav2vec2)   │     │
│                             │   + Classification Head         │     │
│                             └─────────────────────────────────┘     │
│                                                    │                 │
│                                                    ▼                 │
│                             ┌─────────────────────────────────┐     │
│                             │   Softmax + Confidence Scores   │     │
│                             └─────────────────────────────────┘     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
                    Character Predictions with Percentages


┌─────────────────────────────────────────────────────────────────────┐
│                       TRAINING PIPELINE                              │
│                                                                       │
│  characters/              Data Processing           Model Training   │
│  ├── harry/      ───►    ┌──────────────┐    ───►  ┌─────────────┐ │
│  ├── hermione/           │ Load & Split │          │ wav2vec2    │ │
│  ├── ron/                │ Preprocess   │          │ Fine-tuning │ │
│  └── ...                 │ Augment      │          │             │ │
│                          └──────────────┘          └─────────────┘ │
│                                 │                          │         │
│                                 ▼                          ▼         │
│                          ┌──────────────┐          ┌─────────────┐ │
│                          │ DataLoader   │          │ Evaluation  │ │
│                          │ (Batching)   │          │ & Metrics   │ │
│                          └──────────────┘          └─────────────┘ │
│                                                            │         │
│                                                            ▼         │
│                                                     models/best_model.pt │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
audiobook-character-classifier/
├── architecture.md                    # This file
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
├── .gitignore                         # Git ignore rules
│
├── characters/                        # Training data (user-provided)
│   ├── harry/                         # Character 1 audio clips
│   │   ├── clip_001.wav
│   │   ├── clip_002.wav
│   │   └── ...
│   ├── hermione/                      # Character 2 audio clips
│   │   ├── clip_001.wav
│   │   └── ...
│   ├── ron/                           # Character 3 audio clips
│   └── ...                            # Additional characters
│
├── new_samples/                       # Inference data (user-provided)
│   ├── unknown_clip_1.wav
│   ├── unknown_clip_2.wav
│   └── ...
│
├── data/                              # Generated data artifacts
│   ├── splits/                        # Train/val/test splits
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── cache/                         # Preprocessed audio cache
│
├── models/                            # Saved model checkpoints
│   ├── best_model.pt                  # Best model weights
│   ├── checkpoints/                   # Training checkpoints
│   │   ├── epoch_001.pt
│   │   ├── epoch_002.pt
│   │   └── ...
│   └── metadata.json                  # Model metadata (character mapping)
│
├── results/                           # Evaluation results
│   ├── metrics.json                   # Performance metrics
│   ├── confusion_matrix.png           # Confusion matrix visualization
│   ├── training_curves.png            # Loss/accuracy plots
│   └── predictions.csv                # Prediction results
│
├── configs/                           # Configuration files
│   ├── default.yaml                   # Default hyperparameters
│   └── training.yaml                  # Training configuration
│
├── src/                               # Source code
│   ├── __init__.py
│   │
│   ├── data/                          # Data handling modules
│   │   ├── __init__.py
│   │   ├── dataset.py                 # PyTorch Dataset class
│   │   ├── preprocessing.py           # Audio preprocessing
│   │   ├── augmentation.py            # Data augmentation
│   │   └── data_splitter.py           # Train/val/test splitting
│   │
│   ├── models/                        # Model architectures
│   │   ├── __init__.py
│   │   ├── wav2vec2_classifier.py     # Main model architecture
│   │   └── model_utils.py             # Model utilities
│   │
│   ├── training/                      # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Training loop
│   │   └── evaluator.py               # Evaluation logic
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── config.py                  # Configuration management
│       ├── logging_utils.py           # Logging utilities
│       └── visualization.py           # Plotting functions
│
├── scripts/                           # Executable scripts
│   ├── prepare_data.py                # Data preparation
│   ├── train.py                       # Training script
│   ├── evaluate.py                    # Evaluation script
│   └── predict.py                     # Prediction script (MAIN ENTRY)
│
└── notebooks/                         # Jupyter notebooks
    ├── 01_data_exploration.ipynb      # EDA
    ├── 02_model_evaluation.ipynb      # Results analysis
    └── 03_visualizations.ipynb        # Visualizations
```

---

## Data Flow

### Training Phase

1. **Data Discovery**
   - Scan `characters/` directory
   - Identify subdirectories as character labels
   - Enumerate all `.wav` files per character
   - Validate audio files (format, duration, sample rate)

2. **Data Splitting**
   - Stratified split: 70% train, 15% validation, 15% test
   - Ensure balanced class distribution
   - Save splits to `data/splits/*.json`

3. **Preprocessing Pipeline**
   ```
   Raw WAV → Load → Resample to 16kHz → Convert to Mono
           → Trim Silence → Normalize Amplitude → Pad/Trim to Fixed Length
           → Cached Tensor
   ```

4. **Data Augmentation** (Training only)
   - Gaussian noise injection (p=0.7)
   - Time stretching (p=0.4)
   - Gain variation (p=0.5)
   - Applied on-the-fly during training

5. **Feature Extraction**
   - Feed preprocessed audio to wav2vec2 encoder
   - Extract 768-dimensional embeddings
   - Pass through classification head

6. **Training Loop**
   ```python
   for epoch in epochs:
       for batch in train_loader:
           # Forward pass
           embeddings = wav2vec2(audio)
           logits = classifier_head(embeddings)
           loss = cross_entropy(logits, labels)

           # Backward pass
           loss.backward()
           optimizer.step()

       # Validation
       val_metrics = evaluate(model, val_loader)

       # Save best model
       if val_metrics['accuracy'] > best_accuracy:
           save_checkpoint('best_model.pt')
   ```

7. **Model Checkpointing**
   - Save best model based on validation accuracy
   - Save metadata (character names, index mapping)
   - Save training configuration

### Inference Phase

1. **User Request**
   ```bash
   python scripts/predict.py --audio new_samples/unknown_clip.wav
   ```

2. **Model Loading**
   - Load `models/best_model.pt`
   - Load metadata (character mapping)
   - Set model to evaluation mode

3. **Audio Processing**
   - Load audio from `new_samples/` directory
   - Apply same preprocessing as training
   - No augmentation during inference

4. **Prediction**
   ```python
   with torch.no_grad():
       embeddings = wav2vec2(audio)
       logits = classifier_head(embeddings)
       probabilities = softmax(logits)
   ```

5. **Output Generation**
   ```
   Predictions for: unknown_clip.wav
   ─────────────────────────────────────
   1. harry      : 87.32%  ████████████████████
   2. hermione   : 8.45%   ██
   3. ron        : 3.12%   █
   4. dumbledore : 1.11%
   ```

---

## Component Details

### 1. Audio Preprocessing (`src/data/preprocessing.py`)

**Purpose:** Convert raw audio to model-ready tensors

**Key Functions:**
- `load_audio(path)` - Load WAV file with torchaudio
- `resample_audio(waveform, orig_sr, target_sr)` - Resample to 16kHz
- `convert_to_mono(waveform)` - Convert stereo to mono
- `trim_silence(waveform, threshold_db)` - Remove silence
- `normalize_amplitude(waveform, target_db)` - Normalize volume
- `pad_or_trim(waveform, target_length)` - Fixed-length output

**Configuration:**
- Target sample rate: 16,000 Hz (wav2vec2 standard)
- Target duration: 8.0 seconds (configurable)
- Silence threshold: -20 dB
- Normalization target: -3 dB

### 2. Data Augmentation (`src/data/augmentation.py`)

**Purpose:** Increase training data diversity

**Augmentations Applied:**
- Additive Gaussian noise (SNR: 20-40 dB)
- Time stretching (0.9x - 1.1x)
- Pitch shifting (±1 semitone, careful!)
- Gain variation (±6 dB)
- Time shifting (±20%)

**Implementation:** audiomentations library

### 3. Model Architecture (`src/models/wav2vec2_classifier.py`)

**Base Model:** facebook/wav2vec2-base (95M parameters)

**Architecture:**
```python
CharacterVoiceClassifier(
  (wav2vec2): Wav2Vec2Model(
    # Pretrained layers (first 6 frozen)
    # Last 6 layers fine-tuned
  )
  (classifier_head): Sequential(
    Linear(768 → 256)
    BatchNorm1d(256)
    ReLU()
    Dropout(0.3)
    Linear(256 → 128)
    BatchNorm1d(128)
    ReLU()
    Dropout(0.3)
    Linear(128 → num_characters)
  )
)
```

**Training Strategy:**
- Freeze: First 6 transformer layers
- Fine-tune: Last 6 transformer layers + classifier head
- Differential learning rates:
  - Pretrained layers: 1e-5
  - Classifier head: 2e-4

### 4. Training Pipeline (`src/training/trainer.py`)

**Hyperparameters:**
```yaml
optimizer: AdamW
learning_rate: 2e-5
weight_decay: 0.01
batch_size: 16
accumulation_steps: 2  # Effective batch size: 32
epochs: 30
scheduler: CosineAnnealingWarmRestarts
warmup_steps: 100
dropout: 0.3
label_smoothing: 0.1
early_stopping_patience: 5
```

**Loss Function:** CrossEntropyLoss with label smoothing

**Regularization:**
- Dropout in classifier head
- Weight decay (L2 regularization)
- Label smoothing
- Early stopping
- Gradient clipping (max_norm=1.0)

### 5. Evaluation (`src/training/evaluator.py`)

**Metrics Computed:**
- Overall accuracy
- Per-character precision/recall/F1
- Macro F1-score
- Top-3 accuracy
- Confusion matrix

**Visualizations:**
- Training/validation curves
- Confusion matrix heatmap
- Per-character performance bars
- Confidence distribution

### 6. Prediction Script (`scripts/predict.py`)

**Main Entry Point for Users**

**Usage:**
```bash
# Single file prediction
python scripts/predict.py --audio new_samples/clip.wav

# Batch prediction (all files in new_samples/)
python scripts/predict.py --batch

# With confidence threshold
python scripts/predict.py --audio clip.wav --threshold 0.6
```

**Output Format:**
- Character name with confidence percentage
- Visual confidence bar
- Top-K predictions (default: all characters)
- Warning if confidence below threshold

---

## Technology Stack

### Core Dependencies

```
# Deep Learning
torch==2.1.0                    # PyTorch framework
torchaudio==2.1.0               # Audio processing
transformers==4.35.0            # Pretrained models
accelerate==0.24.0              # Training optimization

# Audio Processing
librosa==0.10.1                 # Feature extraction
soundfile==0.12.1               # Audio I/O
audiomentations==0.33.0         # Data augmentation

# ML Utilities
scikit-learn==1.3.2             # Metrics, splits
numpy==1.24.3
pandas==2.1.3

# Visualization
matplotlib==3.8.2
seaborn==0.13.0

# Configuration
pyyaml==6.0.1

# Progress
tqdm==4.66.1
```

---

## Model Training Workflow

### Step 1: Data Preparation
```bash
python scripts/prepare_data.py --characters_dir characters/
```
- Scans character directories
- Creates train/val/test splits
- Validates audio files
- Generates `data/splits/*.json`

### Step 2: Training
```bash
python scripts/train.py --config configs/training.yaml
```
- Loads data splits
- Initializes model
- Runs training loop
- Saves checkpoints
- Logs metrics

### Step 3: Evaluation
```bash
python scripts/evaluate.py --model models/best_model.pt
```
- Loads test set
- Computes metrics
- Generates visualizations
- Saves results

### Step 4: Prediction
```bash
python scripts/predict.py --audio new_samples/unknown.wav
```
- Loads trained model
- Processes audio
- Returns predictions

---

## Configuration Management

### config/default.yaml
```yaml
# Data
data:
  characters_dir: "characters"
  new_samples_dir: "new_samples"
  sample_rate: 16000
  duration: 8.0
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

# Model
model:
  name: "facebook/wav2vec2-base"
  freeze_layers: 6
  dropout: 0.3
  classifier_hidden_dims: [256, 128]

# Training
training:
  batch_size: 16
  epochs: 30
  learning_rate: 2e-5
  weight_decay: 0.01
  early_stopping_patience: 5
  gradient_clip: 1.0

# Augmentation
augmentation:
  noise_prob: 0.7
  time_stretch_prob: 0.4
  gain_prob: 0.5
  pitch_shift_prob: 0.3

# Inference
inference:
  confidence_threshold: 0.5
  top_k: 5
```

---

## Error Handling

### Data Validation
- Check audio file format (WAV required)
- Validate sample rate (auto-resample if needed)
- Verify minimum duration (3 seconds)
- Detect corrupted files
- Check for class imbalance (warn if >3:1 ratio)

### Model Loading
- Verify checkpoint file exists
- Check metadata compatibility
- Validate character mapping
- Handle version mismatches

### Inference
- File not found errors
- Invalid audio format
- Processing failures
- Low confidence warnings

---

## Performance Expectations

### Training Time
- 5 characters, 1,000 clips: ~1-2 hours (GPU)
- 8 characters, 1,600 clips: ~2-3 hours (GPU)
- 10 characters, 2,500 clips: ~3-4 hours (GPU)

### Inference Time
- Single prediction: ~0.5-1 second (GPU)
- Single prediction: ~2-3 seconds (CPU)
- Batch of 10: ~5-10 seconds (GPU)

### Expected Accuracy
- Minimum viable: 60%+ (5 characters)
- Good performance: 75-85% (8 characters)
- Excellent: 85-95% (distinctive voices)

### Memory Requirements
- Training: 8-16 GB GPU RAM
- Inference: 4-8 GB GPU RAM
- CPU inference: 4 GB system RAM

---

## Extensibility

### Adding New Features

1. **Multi-label Classification**
   - Modify output layer for sigmoid activation
   - Support multiple characters per clip

2. **Unknown Character Detection**
   - Implement confidence thresholding
   - Add "unknown" class with negative sampling

3. **Emotion Recognition**
   - Add auxiliary classification head
   - Multi-task learning setup

4. **Real-time Inference**
   - Implement streaming audio processing
   - Sliding window approach

5. **Model Compression**
   - Quantization (INT8)
   - Knowledge distillation
   - ONNX export for deployment

### Alternative Model Architectures

- **HuBERT**: Similar to wav2vec2, slightly different pretraining
- **Whisper-tiny**: Faster but optimized for transcription
- **ECAPA-TDNN**: Efficient speaker verification architecture
- **Custom CNN**: Mel-spectrogram + ResNet backbone

---

## Security & Privacy

### Data Handling
- Audio files never leave local machine
- No external API calls for inference
- Model trained locally

### Model Security
- No telemetry or tracking
- Reproducible results with seeds
- Transparent model architecture

---

## Testing Strategy

### Unit Tests
- Audio preprocessing correctness
- Data loader functionality
- Model forward pass
- Metric calculations

### Integration Tests
- End-to-end training pipeline
- Inference pipeline
- Data preparation workflow

### Validation Tests
- Audio format compatibility
- Different sample rates
- Variable audio lengths
- Edge cases (very short/long clips)

---

## Future Improvements

### Short-term
- [ ] Add progress bars for data preparation
- [ ] Implement model export to ONNX
- [ ] Add confidence calibration
- [ ] Web UI for predictions (Gradio/Streamlit)

### Medium-term
- [ ] Few-shot learning for new characters
- [ ] Active learning for data collection
- [ ] Model ensemble for better accuracy
- [ ] Embedding visualization (t-SNE/UMAP)

### Long-term
- [ ] Multi-language support
- [ ] Real-time streaming inference
- [ ] Mobile deployment (TFLite)
- [ ] API service (FastAPI/Flask)

---

## Troubleshooting

### Common Issues

**Issue:** Out of memory during training
- **Solution:** Reduce batch_size or use gradient accumulation

**Issue:** Poor accuracy (<50%)
- **Solution:** Check data quality, increase dataset size, verify labels

**Issue:** Model overfitting
- **Solution:** Increase dropout, add more augmentation, reduce epochs

**Issue:** Slow inference
- **Solution:** Use GPU, reduce audio duration, optimize preprocessing

**Issue:** Audio file not found
- **Solution:** Verify file path, check new_samples/ directory

---

## References

### Papers
- wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)
- HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units (Hsu et al., 2021)
- ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification (Desplanques et al., 2020)

### Libraries
- PyTorch: https://pytorch.org/
- Transformers: https://huggingface.co/docs/transformers/
- Torchaudio: https://pytorch.org/audio/
- Audiomentations: https://imerit.github.io/audiomentations/

---

## License

This project is designed for educational purposes (academic deep learning course project).

---

## Contact & Support

For implementation questions or issues, refer to:
- Project README.md
- Code comments and docstrings
- Jupyter notebooks for examples

---

**Document Version:** 1.0
**Last Updated:** 2026-03-16
**Author:** Senior ML Engineer
**Status:** Implementation Ready
