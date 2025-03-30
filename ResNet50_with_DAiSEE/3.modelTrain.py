# Tensorflow libraries (Deep Learn tasks)
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy, CategoricalFocalCrossentropy
from tensorflow.keras.metrics import Precision, Recall

# Common Libraries
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from config.config import Config

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
        
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

class EmotionClassifier:
    def __init__(self, data_dir=Config.BALANCE_PROCESSED_DATA_DIR, input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-4, batch_size=32, epochs=30):
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
            
    def prepare_val_class_weights(self):
        # Based on validation set distribution (closer to real-world)
        val_labels = self.val_generator.classes
        unique, counts = np.unique(val_labels, return_counts=True)
        total = len(val_labels)
        
        # Compute weights: less frequent classes get higher weights
        class_weights = {}
        for i, count in enumerate(counts):
            class_weights[i] = total / (len(counts) * count)
            
        print("Class weights:", class_weights)
        return class_weights 
        
    def prepare_data_generators(self):
        train_dir = os.path.join(self.data_dir, "Train")
        val_dir = os.path.join(self.data_dir, "Validation")
        test_dir = os.path.join(self.data_dir, "Test")

        train_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.6, 1.4],
            shear_range=0.2,
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

        # Freeze first 50 layers
        for layer in base_model.layers[:80]:  
            layer.trainable = False  
        for layer in base_model.layers[80:]:  
            layer.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        x = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(x) 
        x = Dropout(0.5)(x)

        x = Dense(128, activation='relu', kernel_regularizer=l2(0.0005))(x)
        x = Dropout(0.4)(x)

        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Fixed decay steps calculation
        total_steps_per_epoch = len(self.train_generator)  # Steps per epoch
        total_training_steps = total_steps_per_epoch * self.epochs  # Total training steps
        warmup_steps = int(0.1 * total_training_steps) # 10% of total steps

        lr_schedule = CosineDecay(
            initial_learning_rate=self.learning_rate, 
            decay_steps=total_steps_per_epoch,  
            alpha=1e-5, 
            warmup_target=1e-4,
            warmup_steps=warmup_steps
        )

        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss=CategoricalFocalCrossentropy(gamma=2.5), 
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), F1Score(name='f1')] 
        )
        

    def train(self, checkpoint_filename='best_model.weights.h5', model_filename='emotion_recognition_model.h5'):
        checkpoint_path = os.path.join(Config.MODEL_DIR, checkpoint_filename)
        model_save_path = os.path.join(Config.MODEL_DIR, model_filename)
        
        if not os.path.exists(Config.MODEL_DIR):
            os.makedirs(Config.MODEL_DIR)

        callbacks = [
            ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True, mode='max', verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
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

        # Save the entire model after training
        self.model.save(model_save_path)
        print(f"Full model saved to {model_save_path}")

        return history


    def evaluate(self, model_filename='emotion_recognition_model.h5'):

        try:
            # Load saved model
            model_path = os.path.join(Config.MODEL_DIR, model_filename)
            model = load_model(model_path)
            print("Model loaded successfully.")

            # Get true labels
            y_true = self.test_generator.classes  # True class labels
            class_names = list(self.test_generator.class_indices.keys())

            # Predict in one go (no manual looping)
            y_pred_probs = model.predict(self.test_generator, verbose=1)
            y_pred = np.argmax(y_pred_probs, axis=1)

            # Classification report
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=class_names))

            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            print("\nConfusion Matrix:")
            print(conf_matrix)

            # Recall per class
            recall_per_class = recall_score(y_true, y_pred, average=None)
            for idx, emotion in enumerate(class_names):
                print(f"Recall ({emotion}): {recall_per_class[idx]:.4f}")

            # Calculate accuracy manually
            accuracy = np.mean(y_true == y_pred)
            print(f"\nTest Accuracy: {accuracy:.4f}")

            # Save evaluation metrics to CSV
            metrics_df = pd.DataFrame(report).transpose()
            metrics_path = os.path.join(Config.MODEL_DIR, 'evaluation_metrics.csv')
            metrics_df.to_csv(metrics_path)
            print(f"Detailed metrics saved to {metrics_path}")

            return {
                'y_true': y_true,
                'y_pred': y_pred,
                'class_names': class_names,
                'report': report,
                'confusion_matrix': conf_matrix,
                'accuracy': accuracy
            }

        except Exception as e:
            print(f"Error during model evaluation: {e}")
            return None

if __name__ == "__main__":
    # Set memory growth to avoid GPU memory errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))
    print("Is TensorFlow using GPU?", tf.test.is_built_with_cuda())
    print("GPU device:", tf.test.gpu_device_name())
        
    classifier = EmotionClassifier()
    classifier.prepare_data_generators()
    classifier.build_model()
    classifier.train()
    classifier.evaluate()