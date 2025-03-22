import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.regularizers import l2
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
from config.config import Config

class EmotionClassifier:
    def __init__(self, data_dir=Config.PROCESSED_DATA_DIR, input_shape=(224, 224, 3), num_classes=4, learning_rate=1e-4, batch_size=32, epochs=30):
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def build_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.4)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
    
    def prepare_data_generators(self):
        train_dir = os.path.join(self.data_dir, "Train")
        val_dir = os.path.join(self.data_dir, "Validation")
        test_dir = os.path.join(self.data_dir, "Test")
        
        # Enhanced data augmentation to help with overfitting
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        print("Loading Data Generators")
        
        # Flow from directory will automatically print "Found X images belonging to Y classes"
        # but we'll also store and print it ourselves for clarity
        self.train_generator = train_datagen.flow_from_directory(
            train_dir, 
            target_size=self.input_shape[:2],
            batch_size=self.batch_size, 
            class_mode='sparse', 
            shuffle=True
        )
        print(f"Train Dataset: Found {self.train_generator.samples} images belonging to {self.train_generator.num_classes} classes.")
        
        self.val_generator = val_datagen.flow_from_directory(
            val_dir, 
            target_size=self.input_shape[:2],
            batch_size=self.batch_size, 
            class_mode='sparse', 
            shuffle=False
        )
        print(f"Validation Dataset: Found {self.val_generator.samples} images belonging to {self.val_generator.num_classes} classes.")
        
        self.test_generator = test_datagen.flow_from_directory(
            test_dir, 
            target_size=self.input_shape[:2],
            batch_size=self.batch_size, 
            class_mode='sparse', 
            shuffle=False
        )
        print(f"Test Dataset: Found {self.test_generator.samples} images belonging to {self.test_generator.num_classes} classes.")
        
        print("Data Generating Complete...")
            
    
    def train(self, checkpoint_filename='best_model.h5'):
        # Combine Config.MODEL_DIR with the filename
        checkpoint_path = os.path.join(Config.MODEL_DIR, checkpoint_filename)
        
        # Make sure the directory exists
        if not os.path.exists(Config.MODEL_DIR):
            os.makedirs(Config.MODEL_DIR)
            
        callbacks = [
            ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
        ]
        
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=len(self.train_generator),
            validation_data=self.val_generator,
            validation_steps=len(self.val_generator),
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
    
        return history
    
    def evaluate(self):
        print("Evaluating model...")
        
        # Get predictions & true labels
        y_true = []
        y_pred = []
        
        steps = len(self.test_generator)
        with tqdm(total=steps, desc="Evaluating") as pbar:
            for i, (X, y) in enumerate(self.test_generator):
                preds = self.model.predict(X, verbose=0)
                y_true.extend(y)
                y_pred.extend(np.argmax(preds, axis=1))
                pbar.update(1)
                
                if i+1 >= steps:
                    break
        
        # Trim to exact size of dataset
        y_true = y_true[:self.test_generator.samples]
        y_pred = y_pred[:self.test_generator.samples]
        
        # Compute Precision & Recall
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        
        # Compute Loss & Accuracy
        loss, accuracy = self.model.evaluate(self.test_generator, verbose=1)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        
        return loss, accuracy, precision, recall
    
    def save_model(self, filepath=os.path.join(Config.MODEL_DIR, 'emotion_recognition_model.h5')):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
if __name__ == "__main__":
    # Set memory growth to avoid GPU memory errors
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized
        pass
        
    classifier = EmotionClassifier()
    classifier.build_model()
    classifier.prepare_data_generators()
    classifier.train()
    classifier.evaluate()
    classifier.save_model()