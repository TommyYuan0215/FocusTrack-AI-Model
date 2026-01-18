# FocusTrack AI Model

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)

A deep learning-based emotion recognition system for detecting student engagement levels using the DAiSEE (Dataset for Affective States in E-Environments) dataset. This project uses ResNet50 transfer learning to classify student emotions into three categories: **Bored**, **Interested**, and **Lacking Focus**.

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
- **Face Detection**: Support for both Haar Cascade (fast) and MTCNN (accurate) face detection
- **Data Balancing**: Intelligent dataset balancing to handle class imbalance
- **Transfer Learning**: ResNet50 pre-trained on ImageNet with custom classification head
- **Two-Phase Training**: Frozen base model training followed by fine-tuning
- **Comprehensive Evaluation**: Detailed metrics including precision, recall, F1-score, and confusion matrix
- **Multi-processing Support**: Parallel video processing for faster preprocessing

## Project Structure 📁

```
FocusTrack-AI-Model/
│
├── config/
│   ├── config.py                          # Configuration settings
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
├── 0.download_daisee.py                   # Dataset download script
├── 1.preprocess_daisee.py                 # Video preprocessing pipeline
├── 2.balance_daisee_dataset.py            # Dataset balancing script
├── 3.modelTrain.py                        # Model training script
├── 4.modelTest.py                         # Model evaluation script
├── main.ipynb                             # Main Jupyter notebook
├── model_train.ipynb                      # Training notebook
├── requirements_resnet50_daisee.txt       # Python dependencies
└── README.md                              # This file
```

## Requirements🔧

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
   python Nvidia_CUDA_CuDNN.py
   ```

## Usage 🚀

### Step 1: Download Dataset

Download the DAiSEE dataset and extract it to the `data/raw` directory:

```bash
python 0.download_daisee.py
```

> **Note**: You'll need to update the download URL in the script with the actual DAiSEE dataset link.

### Step 2: Preprocess Videos

Extract frames from videos and detect faces:

```bash
python 1.preprocess_daisee.py
```

This script will:

- Extract frames from each video at 1-second intervals
- Detect and align faces using Haar Cascade or MTCNN
- Save processed frames to `data/processed/`
- Generate `metadata.csv` with processing statistics

### Step 3: Balance Dataset

Balance the dataset to handle class imbalance:

```bash
python 2.balance_daisee_dataset.py
```

This creates a balanced version of the dataset in `data/balance_processed/`.

### Step 4: Train Model

Train the emotion recognition model:

```bash
python 3.modelTrain.py
```

Training uses a two-phase approach:

- **Phase 1**: Train classification head with frozen ResNet50 base
- **Phase 2**: Fine-tune last 30 layers of ResNet50

### Step 5: Evaluate Model

Evaluate the trained model on the test set:

```bash
python 4.modelTest.py
```

This generates:

- Classification report
- Confusion matrix
- Per-class recall scores
- Overall accuracy
- Detailed metrics CSV

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

The model uses **ResNet50** as the backbone with a custom classification head:

```
Input (224x224x3)
    ↓
Data Augmentation (Random Flip, Rotation)
    ↓
ResNet50 (Pre-trained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense(96) + BatchNorm + ReLU + Dropout(0.25)
    ↓
Dense(64) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(32) + BatchNorm + ReLU + Dropout(0.35)
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

- Freeze ResNet50 base model
- Train only the classification head
- Initial learning rate: 2e-5
- Early stopping with patience=10

### Phase 2: Fine-Tuning (60% of epochs)

- Unfreeze last 30 layers of ResNet50
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
EMOTION_COLUMNS = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
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
