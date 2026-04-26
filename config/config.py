import os


class Config:
    # Base directories
    BASE_DIR = os.path.abspath(os.getcwd())
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')

    # Haar cascade path for face detection
    FACE_CASCADE_PATH = os.path.join(BASE_DIR, 'config', 'haarcascade_frontalface_default.xml')

    # Data directories
    RAW_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    BALANCE_PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'balance_processed')
    ORIGINAL_PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'original_processed')

    # Raw data subdirectories
    LABELS_DIR = os.path.join(RAW_DIR, 'Labels')
    DATASET_DIR = os.path.join(RAW_DIR, 'DataSet')

    # Label files
    LABEL_FILES = {
        'Train': os.path.join(LABELS_DIR, 'TrainLabels.csv'),
        'Validation': os.path.join(LABELS_DIR, 'ValidationLabels.csv'),
        'Test': os.path.join(LABELS_DIR, 'TestLabels.csv'),
    }

    # Emotion columns used for labeling (from DAiSEE dataset)
    EMOTION_COLUMNS = ['Boredom', 'Engagement', 'Confusion', 'Frustration']

    # Model classification targets (mapped from emotion columns)
    NUM_CLASSES = 3
    CLASS_NAMES = {0: 'Bored', 1: 'Interested', 2: 'Lacking_Focus'}

    # Metadata paths
    INPUT_METADATA = os.path.join(PROCESSED_DATA_DIR, 'metadata.csv')
    OUTPUT_METADATA = os.path.join(PROCESSED_DATA_DIR, 'balanced_metadata.csv')

    # Model file names
    MODEL_FILENAME = 'emotion_recognition_model.h5'
    CHECKPOINT_FILENAME = 'best_model.weights.h5'