# FocusTrack AI Model

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)

A deep learning-based emotion recognition system for detecting student engagement levels using the DAiSEE (Dataset for Affective States in E-Environments) dataset. This project uses MobileNetV2 transfer learning to classify student emotions into three categories: **Bored**, **Interested**, and **Lacking Focus**.

## Table of Contents 📋

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Structure](#dataset-structure)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [License](#license)

## Overview 🎯

FocusTrack AI Model is designed to automatically detect and classify student engagement levels from video data. The system processes video frames, detects faces, and uses a fine-tuned ResNet50 model to predict emotional states that indicate engagement levels during learning activities.

## Features ✨

- **Automated Video Processing**: Extract and preprocess frames from video files
- **Face Detection**: Haar Cascade-based face detection and cropping
- **Data Balancing**: Intelligent dataset balancing to handle class imbalance
- **Transfer Learning**: MobileNetV2 pre-trained on ImageNet with custom classification head
- **Two-Phase Training**: Frozen base model training followed by fine-tuning
- **Comprehensive Evaluation**: Detailed metrics including precision, recall, F1-score, and confusion matrix
- **Real-Time Inference**: Webcam-based real-time emotion detection using MediaPipe
- **Multi-processing Support**: Parallel video processing for faster preprocessing

## Project Structure 📁

```
FocusTrack-AI-Model/
│
├── config/
│   ├── config.py                          # Centralized configuration
│   └── haarcascade_frontalface_default.xml # Face detection cascade
│
├── data/
│   ├── raw/                               # Raw dataset
│   │   ├── DataSet/                       # Video files organized by split
│   │   └── Labels/                        # CSV label files
│   │       ├── TrainLabels.csv
│   │       ├── ValidationLabels.csv
│   │       └── TestLabels.csv
│   │
│   ├── processed/                         # Processed frames
│   │   ├── Train/
│   │   ├── Validation/
│   │   ├── Test/
│   │   └── metadata.csv
│   │
│   └── balance_processed/                 # Balanced dataset
│       ├── Train/
│       ├── Validation/
│       ├── Test/
│       └── balanced_metadata.csv
│
├── models/
│   ├── emotion_recognition_model.h5       # Trained model
│   ├── best_model.weights.h5              # Best model weights
│   └── evaluation_metrics.csv             # Model evaluation results
│
├── step1_download.py                      # Step 1: Download DAiSEE dataset
├── step2_preprocess.py                    # Step 2: Video preprocessing pipeline
├── step3_balance.py                       # Step 3: Dataset balancing
├── step4_train.py                         # Step 4: Model training & evaluation
├── step5_inference.py                     # Step 5: Real-time webcam inference
├── check_gpu.py                           # GPU/CUDA verification utility
├── main.ipynb                             # Main Jupyter notebook (with outputs)
├── requirements_resnet50_daisee.txt       # Python dependencies
└── README.md                              # This file
```

## Requirements 🔧

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (for processing large datasets)

See [`requirements_resnet50_daisee.txt`](requirements_resnet50_daisee.txt) for complete dependencies.

## Installation 📦

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/FocusTrack-AI-Model.git
   cd FocusTrack-AI-Model
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements_resnet50_daisee.txt
   ```

4. **Verify CUDA installation** (optional, for GPU support)
   ```bash
   python check_gpu.py
   ```

## Usage 🚀

### Step 1: Download Dataset

Download the DAiSEE dataset and extract it to the `data/raw` directory:

```bash
python step1_download.py
```

> **Note**: You'll need to update the download URL in the script with the actual DAiSEE dataset link.

### Step 2: Preprocess Videos

Extract frames from videos and detect faces:

```bash
python step2_preprocess.py
```

This script will:

- Extract frames from each video at 1-second intervals
- Detect and crop faces using Haar Cascade
- Save processed frames to `data/processed/`
- Generate `metadata.csv` with processing statistics

### Step 3: Balance Dataset

Balance the dataset to handle class imbalance:

```bash
python step3_balance.py
```

This creates a balanced version of the dataset in `data/balance_processed/`.

### Step 4: Train Model

Train the emotion recognition model:

```bash
python step4_train.py
```

Training uses a two-phase approach:

- **Phase 1**: Train classification head with frozen ResNet50 base
- **Phase 2**: Fine-tune last 30 layers of ResNet50

After training completes, the model is automatically evaluated on the test set.

### Step 5: Real-Time Inference

Run real-time emotion detection using your webcam:

```bash
python step5_inference.py
```

This opens a webcam feed with:

- Face detection using MediaPipe
- Emotion prediction with confidence scores
- Press `q` to quit

## Dataset Structure 📊

The DAiSEE dataset should be organized as follows in `data/raw/`:

```
data/raw/
├── DataSet/
│   ├── Train/
│   ├── Validation/
│   └── Test/
└── Labels/
    ├── TrainLabels.csv
    ├── ValidationLabels.csv
    └── TestLabels.csv
```

Each label CSV contains columns:

- `ClipID`: Video file identifier
- `Boredom`: Boredom level (0-4)
- `Engagement`: Engagement level (0-4)
- `Confusion`: Confusion level (0-4)
- `Frustration`: Frustration level (0-4)

## Model Architecture 🧠

The model uses **MobileNetV2** as the backbone with a custom classification head:

```
Input (224x224x3)
    ↓
Data Augmentation (Random Flip, Rotation)
    ↓
MobileNetV2 (Pre-trained on ImageNet, 3.5M params)
    ↓
Global Average Pooling + BatchNorm
    ↓
Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(3, softmax) → [Bored, Interested, Lacking_Focus]
```

**Key Features**:

- L2 regularization on dense layers
- Batch normalization for stable training
- Dropout for preventing overfitting
- Label smoothing (0.1) in loss function
- Cosine decay learning rate schedule

## Training Pipeline 🎓

### Phase 1: Feature Extraction (40% of epochs)

- Freeze MobileNetV2 base model
- Train only the classification head
- Initial learning rate: 2e-5
- Early stopping with patience=10

### Phase 2: Fine-Tuning (60% of epochs)

- Unfreeze last 20 layers of MobileNetV2
- Fine-tune with lower learning rate: 5e-6
- Model checkpoint saves best weights
- Early stopping with patience=10

### Hyperparameters

- **Batch Size**: 32
- **Epochs**: 50 (20 phase 1, 30 phase 2)
- **Optimizer**: Adam with gradient clipping (clipnorm=1.0)
- **Loss**: Categorical Crossentropy with label smoothing
- **Metrics**: Accuracy, Precision, Recall, F1-Score

## Results 📈

After training, evaluation metrics are saved to `models/evaluation_metrics.csv`. Typical results include:

- **Overall Accuracy**: ~75-85%
- **Per-Class Metrics**: Precision, Recall, F1-Score for each emotion
- **Confusion Matrix**: Detailed classification breakdown

## Configuration 📝

All paths and settings are centralized in [`config/config.py`](config/config.py):

```python
# Key configuration options
BASE_DIR = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
NUM_CLASSES = 3
CLASS_NAMES = {0: 'Bored', 1: 'Interested', 2: 'Lacking_Focus'}
```

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- **DAiSEE Dataset**: [Dataset for Affective States in E-Environments](https://iith.ac.in/~daisee-dataset/)
- **ResNet50**: Deep Residual Learning for Image Recognition (He et al., 2015)
- **TensorFlow/Keras**: Deep learning framework

## Contact 📧

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for research and educational purposes. Ensure you have proper permissions to use the DAiSEE dataset according to their terms and conditions.
