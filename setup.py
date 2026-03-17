"""Setup script for audiobook character voice classifier."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="audiobook-character-classifier",
    version="1.0.0",
    author="ML Engineering Team",
    description="Deep learning system for classifying fictional characters from audiobook narration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audiobook-character-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "audiomentations>=0.30.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.20.0",
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "audiobook-prepare=scripts.prepare_data:main",
            "audiobook-train=scripts.train:main",
            "audiobook-evaluate=scripts.evaluate:main",
            "audiobook-predict=scripts.predict:main",
        ],
    },
)
