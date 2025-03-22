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
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    
    # Raw directories inside data directories
    DATASET_DIR = os.path.join(RAW_DIR, 'DataSet')
    LABELS_DIR = os.path.join(RAW_DIR, 'Labels')
    
    # Save the trained model
    VGGFACE_VGG16_PATH = os.path.join(MODEL_DIR, 'vggface_daisee_model_vgg16.h5')
    VGGFACE_RESNET50_PATH = os.path.join(MODEL_DIR, 'vggface_daisee_model_resnet50.h5')
    VGGFACE_SENET50_PATH = os.path.join(MODEL_DIR, 'vggface_daisee_model_senet50.h5')

    # Label files
    LABEL_FILES = {
        'Train': os.path.join(LABELS_DIR, 'TrainLabels.csv'),
        'Validation': os.path.join(LABELS_DIR, 'ValidationLabels.csv'),
        'Test': os.path.join(LABELS_DIR, 'TestLabels.csv')
    }

    # Emotion columns for labeling
    EMOTION_COLUMNS = ['Boredom', 'Engagement', 'Confusion', 'Frustration']