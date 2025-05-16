import os

class Config:
    # Base directories
    BASE_DIR = os.path.abspath(os.getcwd())
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # Haar cascade path for face detection
    CASCADE_PATH = os.path.join(BASE_DIR, 'config/haarcascade_frontalface_default.xml')
    
    # Data directories (with Raw Data and Processed Data)
    RAW_DIR = os.path.join(DATA_DIR, 'raw')
    ORIGINAL_PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'original_processed')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    BALANCE_PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'balance_processed')
    
    # Raw directories inside data directories
    LABELS_DIR = os.path.join(DATA_DIR, 'raw/Labels')
    DATASET_DIR = os.path.join(DATA_DIR, 'raw/DataSet')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    BALANCED_DIR = os.path.join(DATA_DIR, "balance_processed")
    
    PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

    # Label files
    LABEL_FILES = {
        'Train': os.path.join(LABELS_DIR, 'TrainLabels.csv'),
        'Validation': os.path.join(LABELS_DIR, 'ValidationLabels.csv'),
        'Test': os.path.join(LABELS_DIR, 'TestLabels.csv')
    }

    # Emotion columns for labeling
    EMOTION_COLUMNS = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
    
    # Metadata input and output that store in processed dir
    INPUT_METADATA = os.path.join(PROCESSED_DATA_DIR, 'metadata.csv')
    OUTPUT_METADATA = os.path.join(PROCESSED_DATA_DIR, 'balanced_metadata.csv')
    
    # Paths that store in models
    EMOTIONAL_RECOGNITION_MODEL = os.path.join(MODEL_DIR, 'emotional_recognition_model.h5')