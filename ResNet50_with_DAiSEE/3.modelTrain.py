import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy, CategoricalFocalCrossentropy
from tensorflow.keras import mixed_precision

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from config.config import Config

class EmotionClassifier:
    def __init__(self, data_dir=Config.BALANCE_PROCESSED_DATA_DIR, input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-5, batch_size=32, epochs=30):
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
        
    def print_class_distribution(self, labels): 
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels) 
        print("Class Distribution:") 
        for cls, count in zip(unique, counts): 
            percentage = (count / total) * 100 
            print(f"Class {cls}: {count} samples ({percentage:.2f}%)")
        
    def prepare_data_generators(self):
        train_dir = os.path.join(self.data_dir, "Train")
        val_dir = os.path.join(self.data_dir, "Validation")
        test_dir = os.path.join(self.data_dir, "Test")

        train_datagen = ImageDataGenerator(
            rotation_range=10,  
            width_shift_range=0.1,  
            height_shift_range=0.1,  
            zoom_range=0.1,  
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],  
            shear_range=0.1,  
            fill_mode='nearest',  
            preprocessing_function=preprocess_input
        )

        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.train_generator = train_datagen.flow_from_directory(
            train_dir, 
            target_size=self.input_shape[:2], 
            batch_size=self.batch_size, 
            class_mode='categorical', 
            shuffle=True
        ) 

        self.val_generator = val_datagen.flow_from_directory(
            val_dir, target_size=self.input_shape[:2], 
            batch_size=self.batch_size, 
            class_mode='categorical', 
            shuffle=False
        )

        self.test_generator = test_datagen.flow_from_directory(
            test_dir, 
            target_size=self.input_shape[:2], 
            batch_size=self.batch_size, 
            class_mode='categorical', 
            shuffle=False
        )
        
        # Print class distribution for train generator
        print("Training Data Class Distribution:")
        train_labels = self.train_generator.classes
        self.print_class_distribution(train_labels)

        # Optionally, print for validation and test generators
        print("\nValidation Data Class Distribution:")
        val_labels = self.val_generator.classes
        self.print_class_distribution(val_labels)

        print("\nTest Data Class Distribution:")
        test_labels = self.test_generator.classes
        self.print_class_distribution(test_labels)

    def build_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze first 80 layer to perform training
        for layer in base_model.layers[:80]:  
            layer.trainable = False  
        for layer in base_model.layers[80:]:  
            layer.trainable = True
            
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        x = Dense(256, kernel_regularizer=l2(0.0001))(x)
        x = BatchNormalization()(x)  
        x = tf.keras.layers.ReLU()(x)
        x = Dropout(0.5)(x)

        x = Dense(128, kernel_regularizer=l2(0.0001))(x)
        x = BatchNormalization()(x) 
        x = tf.keras.layers.ReLU()(x)
        x = Dropout(0.4)(x)

        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Decay steps modification
        total_steps_per_epoch = len(self.train_generator)  # Total training batches
        decay_steps = total_steps_per_epoch * self.epochs

        lr_schedule = CosineDecay(initial_learning_rate=self.learning_rate, decay_steps=decay_steps, alpha=1e-6)

        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(
            optimizer = Adam(learning_rate=lr_schedule),
            loss = CategoricalFocalCrossentropy(gamma=2),  
            metrics = ['accuracy', Precision(name='precision'), Recall(name='recall')]
        )
        

    def train(self, checkpoint_filename='best_model.weights.h5'):
        checkpoint_path = os.path.join(Config.MODEL_DIR, checkpoint_filename)
        
        if not os.path.exists(Config.MODEL_DIR):
            os.makedirs(Config.MODEL_DIR)

        callbacks = [
            ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True, mode='max', verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
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
        y_true, y_pred = [], []

        steps = len(self.test_generator)
        for i, (X, y) in tqdm(enumerate(self.test_generator), total=steps, desc="Evaluating"):
            preds = self.model.predict(X, verbose=0)
            y_true.extend(np.argmax(y, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
            if i+1 >= steps:
                break

        y_true = np.array(y_true[:len(y_pred)])
        y_pred = np.array(y_pred[:len(y_true)])

        print("\nClassification Report:")
        class_names = list(self.test_generator.class_indices.keys())
        print(classification_report(y_true, y_pred, target_names=class_names))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        recall_per_class = recall_score(y_true, y_pred, average=None)
        for idx, emotion in enumerate(class_names):
            print(f"Recall ({emotion}): {recall_per_class[idx]:.4f}")

        loss, accuracy = self.model.evaluate(self.test_generator, verbose=1)
        print(f"\nTest Accuracy: {accuracy:.4f}")
    
    def save_model(self, filepath=os.path.join(Config.MODEL_DIR, 'emotion_recognition_model.h5')):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

# Set mixed policies        
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

if __name__ == "__main__":
    # Set memory growth to avoid GPU memory errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        
    classifier = EmotionClassifier()
    classifier.prepare_data_generators()
    classifier.build_model()
    classifier.train()
    classifier.evaluate()
    classifier.save_model()