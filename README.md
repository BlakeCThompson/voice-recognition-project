# Audiobook Character Voice Classification

A deep learning system that classifies fictional characters from audiobook narration based on voice characteristics. Uses transfer learning with wav2vec2 pretrained speech models to achieve high accuracy with limited training data.

## Features

- **Transfer Learning**: Fine-tunes facebook/wav2vec2-base for character voice classification
- **Easy to Use**: Simple command-line interface for training and prediction
- **Robust Preprocessing**: Automatic audio normalization, silence trimming, and resampling
- **Data Augmentation**: Multiple augmentation strategies to improve generalization
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualizations
- **Confidence Scores**: Provides probability distributions across all characters

## Project Structure

```
audiobook-character-classifier/
├── architecture.md           # Detailed system architecture documentation
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── configs/
│   └── default.yaml          # Configuration file
├── characters/               # Training data (you provide this)
│   ├── harry/
│   │   ├── clip_001.wav
│   │   └── ...
│   ├── hermione/
│   └── ...
├── new_samples/              # New audio files to classify
│   └── unknown_clip.wav
├── scripts/
│   ├── prepare_data.py       # Data preparation
│   ├── train.py              # Model training
│   ├── evaluate.py           # Model evaluation
│   └── predict.py            # Prediction (main entry point)
├── src/                      # Source code
└── models/                   # Saved models
```

## Quick Start

### 1. Installation

```bash
# Clone or download this repository
cd audiobook-character-classifier

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Organize your audio files in the following structure:

```
characters/
├── character1/
│   ├── clip_001.wav
│   ├── clip_002.wav
│   └── ...
├── character2/
│   ├── clip_001.wav
│   └── ...
└── ...
```

**Requirements:**
- Each character should have their own subdirectory
- Audio files should be WAV format (MP3, FLAC, OGG also supported)
- Recommended: 150-250 clips per character, 4-8 seconds each
- Minimum: 50 clips per character, 3+ seconds each

### 3. Prepare Data Splits

```bash
python scripts/prepare_data.py
```

This will:
- Scan the `characters/` directory
- Validate audio files
- Create train/validation/test splits (70/15/15%)
- Save splits to `data/splits/`

### 4. Train Model

```bash
python scripts/train.py
```

Training will:
- Load pretrained wav2vec2 model
- Fine-tune on your character data
- Save checkpoints to `models/`
- Generate training curves

**Training time:** 1-3 hours on GPU (depending on dataset size)

### 5. Evaluate Model

```bash
python scripts/evaluate.py
```

Evaluation generates:
- Overall accuracy and F1 scores
- Per-character metrics
- Confusion matrix
- Visualizations in `results/`

### 6. Make Predictions

**Single file prediction:**

```bash
python scripts/predict.py --audio new_samples/unknown_clip.wav
```

**Batch prediction (all files in new_samples/):**

```bash
python scripts/predict.py --batch
```

**With options:**

```bash
# Show only top 3 predictions
python scripts/predict.py --audio clip.wav --top_k 3

# Custom confidence threshold
python scripts/predict.py --audio clip.wav --threshold 0.7
```

## Example Output

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
==================================================================
```

## Configuration

Edit `configs/default.yaml` to customize:

- **Data settings**: Sample rate, duration, split ratios
- **Model settings**: Architecture, dropout, frozen layers
- **Training settings**: Batch size, learning rate, epochs
- **Augmentation**: Types and probabilities of augmentations

## Advanced Usage

### Resume Training

```bash
python scripts/train.py --resume models/checkpoints/epoch_010.pt
```

### Evaluate on Different Split

```bash
python scripts/evaluate.py --split val  # or train, test
```

### Custom Configuration

```bash
python scripts/train.py --config configs/custom.yaml
```

## Model Architecture

- **Base Model**: facebook/wav2vec2-base (95M parameters)
- **Strategy**: Transfer learning with layer freezing
  - Freeze: First 6 transformer layers
  - Fine-tune: Last 6 transformer layers + classification head
- **Classification Head**:
  - Linear(768 → 256) + BatchNorm + ReLU + Dropout
  - Linear(256 → 128) + BatchNorm + ReLU + Dropout
  - Linear(128 → num_characters)

## Performance Expectations

### Expected Accuracy
- 5 characters, 750 clips: 60-75%
- 8 characters, 1,600 clips: 75-85%
- 10 characters, 2,500 clips: 80-90%

### Hardware Requirements
- **Training**: 8-16 GB GPU RAM (or CPU with patience)
- **Inference**: 4 GB RAM
- **Storage**: ~500 MB per 1,000 audio clips

### Training Time
- 5 characters, 1,000 clips: ~1-2 hours (GPU)
- 8 characters, 1,600 clips: ~2-3 hours (GPU)
- CPU training: 5-10x slower

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size in configs/default.yaml
training:
  batch_size: 8  # instead of 16
```

### Audio Loading Error
- Ensure audio files are valid WAV format
- Check file permissions
- Verify minimum duration (3 seconds)

### Poor Accuracy
- Collect more data per character (aim for 150+ clips)
- Ensure clips are 4-8 seconds long
- Verify character labels are correct
- Check that characters have distinct voices

### Model Not Found
```bash
# Train a model first
python scripts/train.py
```

## Data Collection Tips

1. **Use distinct voices**: Choose characters with clearly different voice characteristics
2. **Consistent quality**: Use clips from the same audiobook/narrator
3. **Vary context**: Include different emotions, speaking rates, dialogue types
4. **Minimum length**: 4+ seconds gives better results than 2-3 seconds
5. **Balance classes**: Try to have similar number of clips per character

## Citation

If you use this project for academic work, please cite:

```
@software{audiobook_character_classifier,
  title = {Audiobook Character Voice Classification},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/audiobook-character-classifier}
}
```

## References

- **wav2vec 2.0**: Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", NeurIPS 2020
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/
- **PyTorch**: https://pytorch.org/

## License

This project is designed for educational purposes. See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For questions or issues:
1. Check the [architecture.md](architecture.md) for detailed documentation
2. Review the troubleshooting section above
3. Open an issue on GitHub

## Acknowledgments

- HuggingFace for pretrained models
- PyTorch and torchaudio teams
- AudioMentations library
- The open-source ML community

---

**Status**: Production Ready
**Version**: 1.0.0
**Last Updated**: 2024-03-16
