import os

class Config:
    # Base directories
    BASE_DIR = os.path.abspath(os.getcwd())
    RESNET50_DIR = os.path.join(BASE_DIR, 'ResNet50_with_DAiSEE')
    DATA_DIR = os.path.join(RESNET50_DIR, 'data')
    MODEL_DIR = os.path.join(RESNET50_DIR, 'models')
    
    # Haar cascade path for face detection
    FACE_CASCADE_PATH = os.path.join(RESNET50_DIR, 'config', 'haarcascade_frontalface_default.xml')
    
    # Data directories (with Raw Data and Processed Data)
    RAW_DIR = os.path.join(DATA_DIR, 'raw')
    ORIGINAL_PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'original_processed')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    BALANCE_PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'balance_processed')
    
    # Raw directories inside data directories
    DATASET_DIR = os.path.join(RAW_DIR, 'DataSet')
    LABELS_DIR = os.path.join(RAW_DIR, 'Labels')

    # Label files
    LABEL_FILES = {
        'Train': os.path.join(LABELS_DIR, 'TrainLabels.csv'),
        'Validation': os.path.join(LABELS_DIR, 'ValidationLabels.csv'),
        'Test': os.path.join(LABELS_DIR, 'TestLabels.csv')
    }

    # Emotion columns for labeling
    EMOTION_COLUMNS = ['Boredom', 'Engagement', 'Confusion', 'Frustration']