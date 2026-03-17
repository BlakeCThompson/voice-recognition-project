# Quick Start Guide

Get up and running in 5 minutes!

## 1. Install (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt
```

## 2. Prepare Data (1 minute)

Organize your audio files:

```
characters/
├── character1/
│   ├── clip_001.wav
│   └── clip_002.wav
├── character2/
│   └── clip_001.wav
└── ...
```

Then run:

```bash
python scripts/prepare_data.py
```

## 3. Train (1-3 hours)

```bash
python scripts/train.py
```

Takes 1-3 hours depending on dataset size and hardware.

## 4. Predict (30 seconds)

Place audio file in `new_samples/` directory, then:

```bash
python scripts/predict.py --audio new_samples/unknown.wav
```

## Example Output

```
==================================================================
PREDICTION RESULT
==================================================================

Audio File: unknown.wav

Predicted Character: harry
Confidence: 87.32%

All Character Probabilities:
------------------------------------------------------------------
1. harry      :  87.32%  ██████████████████████████████████████████
2. hermione   :   8.45%  ████
3. ron        :   3.12%  █
==================================================================
```

## Common Commands

```bash
# Prepare data
python scripts/prepare_data.py

# Train model
python scripts/train.py

# Evaluate model
python scripts/evaluate.py

# Predict single file
python scripts/predict.py --audio new_samples/clip.wav

# Predict all files
python scripts/predict.py --batch

# Show top 3 predictions only
python scripts/predict.py --audio clip.wav --top_k 3
```

## Minimum Requirements

- **Python**: 3.8+
- **RAM**: 8GB
- **Data**: 50+ audio clips per character, 3+ seconds each
- **GPU**: Optional but recommended for training

## Need Help?

- Full documentation: [README.md](README.md)
- Detailed guide: [USAGE_GUIDE.md](USAGE_GUIDE.md)
- Architecture details: [architecture.md](architecture.md)

## Project Structure

```
audiobook-character-classifier/
├── characters/        # Your training data goes here
├── new_samples/       # Files to predict go here
├── scripts/           # Executable scripts
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── configs/           # Configuration files
├── models/            # Saved models (created during training)
└── results/           # Evaluation results
```

That's it! 🚀
