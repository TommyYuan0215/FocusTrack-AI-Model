import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalFocalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing import image_dataset_from_directory

import os
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
from config.config import Config

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

class EmotionClassifier:
    def __init__(self, data_dir=Config.BALANCE_PROCESSED_DATA_DIR, input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-5, batch_size=32, epochs=50):
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    # Helpoer function to print class distribution
    def print_class_distribution(self):
        def count_class_samples(dataset, dataset_name):
            class_counts = np.zeros(self.num_classes, dtype=int)
            total_samples = 0

            # Convert dataset to a NumPy array (reduces computation overhead)
            for _, y in dataset:
                y_indices = tf.argmax(y, axis=1).numpy()  # Convert one-hot to class indices
                unique, counts = np.unique(y_indices, return_counts=True)
                
                for cls, count in zip(unique, counts):
                    class_counts[cls] += count
                
                total_samples += len(y_indices)

            print(f"\n{dataset_name} Data Class Distribution:")
            for i, count in enumerate(class_counts):
                percentage = (count / total_samples) * 100
                print(f"Class {i}: {count} samples ({percentage:.2f}%)")

        count_class_samples(self.train_dataset, "Training")
        count_class_samples(self.val_dataset, "Validation")
        count_class_samples(self.test_dataset, "Test")
    
    # Get class weight so that it can be priortize minority classes  
    def get_class_weights(self, dataset):
        # Extract all labels from the dataset
        all_labels = []
        for _, labels in dataset:
            # Convert one-hot encoded labels to class indices
            label_indices = tf.argmax(labels, axis=1).numpy()
            all_labels.extend(label_indices)
        
        # Convert to numpy array
        all_labels = np.array(all_labels)
        
        # Get unique classes
        classes = np.unique(all_labels)
        
        # Compute balanced weights using scikit-learn
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=all_labels
        )
        
        # Create dictionary mapping class indices to weights
        class_weights = {i: weight for i, weight in enumerate(weights)}
        
        print(f"Computed class weights: {class_weights}")
        return class_weights

    def load_data(self, input_shape=(224, 224), batch_size=32):
        train_dir = os.path.join(self.data_dir, "Train")
        val_dir = os.path.join(self.data_dir, "Validation")
        test_dir = os.path.join(self.data_dir, "Test")
        
        # More aggressive data augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),              
            tf.keras.layers.RandomZoom(0.2),                  
            tf.keras.layers.RandomContrast(0.2),             
            tf.keras.layers.RandomBrightness(0.2),            
            tf.keras.layers.RandomTranslation(0.1, 0.1)       
        ])
        
        # The rest remains the same
        self.train_dataset = image_dataset_from_directory(
            train_dir,
            image_size=input_shape,
            batch_size=batch_size,
            label_mode="categorical"
        ).map(
            lambda x, y: (data_augmentation(x, training=True), y) 
        ).map(
            lambda x, y: (preprocess_input(x), y) 
        ).prefetch(tf.data.AUTOTUNE)

        self.val_dataset = image_dataset_from_directory(
            val_dir,
            image_size=input_shape,
            batch_size=batch_size,
            label_mode="categorical"
        ).map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)

        self.test_dataset = image_dataset_from_directory(
            test_dir,
            image_size=input_shape,
            batch_size=batch_size,
            label_mode="categorical"
        ).map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)

        print("Data loaded successfully!")
    
        return self.train_dataset, self.val_dataset, self.test_dataset


    def build_model(self):
        # Initialize ResNet50 with ImageNet weights
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Initially freeze the entire base model
        base_model.trainable = False
        
        # Create model architecture
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Enhanced feature extraction head
        x = Dense(128, kernel_regularizer=l2(0.001))(x)       
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)                                    
        
        x = Dense(64, kernel_regularizer=l2(0.001))(x)         
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)                                    
        
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create and compile the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Use a moderate learning rate to start
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), F1Score(name='f1')]
        )
        
        # After some initial training, you can unfreeze some layers
        # Unfreeze the last 50 layers of the ResNet model
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        for layer in base_model.layers[-50:]:
            layer.trainable = True
        
        # Recompile with a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), F1Score(name='f1')]
        )
        
        return self.model

    def train(self, checkpoint_filename='best_model.weights.h5', model_filename='emotion_recognition_model.h5'):
        checkpoint_path = os.path.join(Config.MODEL_DIR, checkpoint_filename)
        model_save_path = os.path.join(Config.MODEL_DIR, model_filename)

        if not os.path.exists(Config.MODEL_DIR):
            os.makedirs(Config.MODEL_DIR)
            
        # Calculate class weights
        class_weights = self.get_class_weights(self.train_dataset)
        
        # Phase 1: Train only the top layers (base model is frozen)
        print("Phase 1: Training with frozen base model...")
        callbacks_phase1 = [
            EarlyStopping(monitor='val_f1', patience=5, mode='max', restore_best_weights=True, verbose=1)
        ]
        
        history1 = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=15,
            callbacks=callbacks_phase1,
            class_weight=class_weights,
            verbose=1
        )
        
        # Phase 2: Fine-tune with gradually unfrozen layers
        print("Phase 2: Fine-tuning with partially unfrozen model...")
        
        # Get the base model (first layer in the model)
        base_model = self.model.layers[0]
        
        # Unfreeze only the last 20 layers (reduced from 50)
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        for layer in base_model.layers[-20:]:
            layer.trainable = True
        
        # Recompile with a much lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=1e-6),  # Reduced from 1e-5
            loss=CategoricalCrossentropy(),
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), F1Score(name='f1')]
        )
        
        # Continue training with fine-tuning
        callbacks_phase2 = [
            ModelCheckpoint(checkpoint_path, monitor='val_f1', save_best_only=True, save_weights_only=True, 
                        mode='max', verbose=1),
            EarlyStopping(monitor='val_f1', patience=10, mode='max', restore_best_weights=True, verbose=1)
        ]
        
        history2 = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=35,
            callbacks=callbacks_phase2,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save the final model
        self.model.save(model_save_path)
        print(f"Full model saved to {model_save_path}")
        
        # Combine histories for returning
        combined_history = {}
        for key in history1.history.keys():
            combined_history[key] = history1.history[key] + history2.history[key]
        
        return combined_history

    def evaluate(self, model_filename='emotion_recognition_model.h5'):
        model_path = os.path.join(Config.MODEL_DIR, model_filename)
        model = load_model(model_path)
        print("Model loaded successfully.")

        # Extract ground truth labels
        y_true = np.concatenate([y.numpy() for _, y in self.test_dataset], axis=0)

        # Define class names in order
        class_names = {0: "Bored", 1: "Interested", 2: "Lacking_Focus"}

        # Get model predictions
        y_pred_probs = model.predict(self.test_dataset, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)  # Convert softmax probabilities to class indices

        # Ensure y_true is in the correct format
        if y_true.ndim > 1:  # If y_true is one-hot encoded
            y_true = np.argmax(y_true, axis=1)

        # Classification Report
        report = classification_report(y_true, y_pred, target_names=list(class_names.values()), output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=list(class_names.values())))

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(conf_matrix)

        # Recall Score for Each Class
        recall_per_class = recall_score(y_true, y_pred, average=None)
        for idx, emotion in class_names.items():
            print(f"Recall ({class_names[idx]}): {recall_per_class[idx]:.4f}")

        # Calculate Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")

        # Save Classification Report to CSV
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

if __name__ == "__main__":
    classifier = EmotionClassifier()
    classifier.load_data()
    # classifier.print_class_distribution()
    classifier.build_model()
    classifier.train()
    classifier.evaluate()
