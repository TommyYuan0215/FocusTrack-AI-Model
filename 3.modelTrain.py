import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation, BatchNormalization, Conv2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Sequential

import os
import numpy as np
import pandas as pd
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
    def __init__(self, 
                 data_dir=Config.BALANCE_PROCESSED_DATA_DIR, 
                 input_shape=(224, 224, 3), 
                 num_classes=3, 
                 batch_size=32, 
                 epochs=50,
                 initial_lr_phase1=2e-5,
                 initial_lr_phase2=5e-6,
                 lr_decay_alpha=1e-6 # Alpha for CosineDecay
                 ):
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.base_model = None
        self.class_weights = None

        # Learning Rate Hyperparameters
        self.initial_lr_phase1 = initial_lr_phase1
        self.initial_lr_phase2 = initial_lr_phase2
        self.lr_decay_alpha_phase1 = lr_decay_alpha 
        self.lr_decay_alpha_phase2 = lr_decay_alpha / 10
    
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
                percentage = (count / total_samples) * 100 if total_samples > 0 else 0
                print(f"Class {i}: {count} samples ({percentage:.2f}%)")

        if self.train_dataset: count_class_samples(self.train_dataset, "Training")
        if self.val_dataset: count_class_samples(self.val_dataset, "Validation")
        if self.test_dataset: count_class_samples(self.test_dataset, "Test")
    
    def load_data(self):
        input_shape = (self.input_shape[0], self.input_shape[1])
        batch_size = self.batch_size

        train_dir = os.path.join(self.data_dir, "Train")
        val_dir = os.path.join(self.data_dir, "Validation")
        test_dir = os.path.join(self.data_dir, "Test")

        self.train_dataset = image_dataset_from_directory(
            train_dir,
            image_size=input_shape,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=True
        ).prefetch(tf.data.AUTOTUNE)

        self.val_dataset = image_dataset_from_directory(
            val_dir,
            image_size=input_shape,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=False
        ).prefetch(tf.data.AUTOTUNE)

        self.test_dataset = image_dataset_from_directory(
            test_dir,
            image_size=input_shape,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=False
        ).prefetch(tf.data.AUTOTUNE)

        print("Data loaded successfully!")
        return self.train_dataset, self.val_dataset, self.test_dataset

    def build_model(self):
        self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        self.base_model.trainable = False
        
        inputs = Input(shape=self.input_shape)
        
        augmentation_pipeline = Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
        ], name="augmentation_pipeline")
        
        x = augmentation_pipeline(inputs)
        x = preprocess_input(x)
        x = self.base_model(x, training=False) 
        
        # Modified Head with three Dense blocks (128, 96, 64)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.15)(x)  # Dropout after pooling

        x = Dense(96, kernel_regularizer=l2(0.0003), kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)

        x = Dense(64, kernel_regularizer=l2(0.0004), kernel_initializer='he_normal')(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x = Dense(32, kernel_regularizer=l2(0.0005), kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.35)(x)
        
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=predictions)
        
        # Initial compile, will be recompiled in _train_phase
        self.model.compile(
            optimizer=Adam(learning_rate=self.initial_lr_phase1), # Placeholder LR
            loss=CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), F1Score(name='f1')]
        )
        return self.model

    def _get_num_batches_per_epoch(self):
        """Determines the number of batches per epoch for the training dataset."""
        num_batches_per_epoch_tensor = tf.data.experimental.cardinality(self.train_dataset)
        if num_batches_per_epoch_tensor < 0: 
            print("Warning: Training dataset cardinality is unknown or infinite. Attempting to infer steps_per_epoch.")
            # This fallback is inefficient and should ideally be avoided by ensuring
            # the dataset (e.g., from image_dataset_from_directory) has a known cardinality.
            try:
                num_total_samples = 0
                for _ in self.train_dataset.unbatch().batch(1): 
                    num_total_samples +=1
                if num_total_samples == 0:
                    raise ValueError("Could not determine number of samples from the dataset.")
                num_batches_per_epoch = (num_total_samples + self.batch_size - 1) // self.batch_size
                print(f"Inferred total samples: {num_total_samples}, Inferred batches per epoch: {num_batches_per_epoch}")
            except Exception as e:
                raise ValueError(
                    "Dataset cardinality is unknown/infinite and could not be inferred. "
                    "Ensure your training dataset has a known size or provide steps_per_epoch. Error: {}".format(e)
                )
        else:
            num_batches_per_epoch = num_batches_per_epoch_tensor.numpy()
        
        if num_batches_per_epoch == 0:
            raise ValueError("Number of batches per epoch is zero. Check dataset and batch size.")
        print(f"Actual batches per epoch: {num_batches_per_epoch}")
        return num_batches_per_epoch

    def _train_phase(self, phase_name, epochs, initial_lr, lr_decay_alpha, callbacks_list, unfreeze_layers_count=None):
        """Helper function to run a training phase."""
        print(f"\n{phase_name}: Training for {epochs} epochs...")

        if unfreeze_layers_count is not None and self.base_model:
            print(f"Unfreezing last {unfreeze_layers_count} layers of the base model.")
            for layer in self.base_model.layers: # Freeze all first
                layer.trainable = False
            for layer in self.base_model.layers[-unfreeze_layers_count:]:
                layer.trainable = True
        elif self.base_model: # If unfreeze_layers_count is None, ensure base_model is frozen (for phase 1)
             self.base_model.trainable = False


        num_batches_per_epoch = self._get_num_batches_per_epoch()
        decay_steps = epochs * num_batches_per_epoch
        print(f"{phase_name} decay steps: {decay_steps}")

        lr_schedule = CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=lr_decay_alpha
        )

        self.model.compile(
            optimizer=Adam(learning_rate=lr_schedule, clipnorm=1.0),
            loss=CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), F1Score(name='f1')]
        )

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1
        )
        return history

    def train(self, checkpoint_filename='best_model.weights.h5', model_filename='emotion_recognition_model.h5'):
        checkpoint_path = os.path.join(Config.MODEL_DIR, checkpoint_filename)
        model_save_path = os.path.join(Config.MODEL_DIR, model_filename)

        if not os.path.exists(Config.MODEL_DIR):
            os.makedirs(Config.MODEL_DIR)
            
        phase1_epochs = int(self.epochs * 0.4)
        phase2_epochs = self.epochs - phase1_epochs
        
        # Callbacks for Phase 1
        callbacks_phase1 = [
            EarlyStopping(
                monitor='val_f1', patience=10, mode='max', # Hardcoded patience
                restore_best_weights=True, verbose=1, min_delta=0.001
            )
        ]
        
        history1 = self._train_phase(
            phase_name="Phase 1 (Frozen Base Model)",
            epochs=phase1_epochs,
            initial_lr=self.initial_lr_phase1,
            lr_decay_alpha=self.lr_decay_alpha_phase1,
            callbacks_list=callbacks_phase1
        )
        
        # Callbacks for Phase 2
        callbacks_phase2 = [
            ModelCheckpoint(
                checkpoint_path, monitor='val_f1', save_best_only=True,
                save_weights_only=True, mode='max', verbose=1
            ),
            EarlyStopping(
                monitor='val_f1', patience=10, mode='max', # Hardcoded patience
                restore_best_weights=True, verbose=1, min_delta=0.001
            )
        ]

        history2 = self._train_phase(
            phase_name="Phase 2 (Fine-tuning)",
            epochs=phase2_epochs,
            initial_lr=self.initial_lr_phase2,
            lr_decay_alpha=self.lr_decay_alpha_phase2,
            callbacks_list=callbacks_phase2,
            unfreeze_layers_count=30
        )
        
        self.model.save(model_save_path)
        print(f"Full model saved to {model_save_path}")
        
        combined_history = {}
        for key in history1.history.keys():
            combined_history[key] = history1.history[key] + history2.history.get(key, [])
        for key in history2.history.keys(): # Add any keys unique to history2
            if key not in combined_history:
                 combined_history[key] = history1.history.get(key, []) + history2.history[key]

        return combined_history

    def evaluate(self, model_filename='emotion_recognition_model.h5'):
        model_path = os.path.join(Config.MODEL_DIR, model_filename)
        # Ensure the custom F1Score is available for loading
        custom_objects = {'F1Score': F1Score}
        try:
            model = load_model(model_path, custom_objects=custom_objects)
            print("Model loaded successfully with custom objects.")
        except Exception as e:
            print(f"Error loading model with custom objects: {e}. Trying without.")
            model = load_model(model_path) # Fallback if F1Score wasn't compiled in saved model
            print("Model loaded successfully (fallback).")


        y_true_list = []
        for _, y_batch in self.test_dataset:
            y_true_list.append(y_batch.numpy())
        
        if not y_true_list:
            print("Test dataset is empty or could not be iterated.")
            return {}
            
        y_true = np.concatenate(y_true_list, axis=0)

        class_names = {0: "Bored", 1: "Interested", 2: "Lacking_Focus"}

        y_pred_probs = model.predict(self.test_dataset, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)

        report = classification_report(y_true, y_pred, target_names=list(class_names.values()), output_dict=True, zero_division=0)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=list(class_names.values()), zero_division=0))

        conf_matrix = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(conf_matrix)

        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        for idx, emotion in class_names.items():
            if idx < len(recall_per_class):
                 print(f"Recall ({class_names[idx]}): {recall_per_class[idx]:.4f}")

        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")

        metrics_df = pd.DataFrame(report).transpose()
        metrics_path = os.path.join(Config.MODEL_DIR, 'evaluation_metrics.csv')
        metrics_df.to_csv(metrics_path)
        print(f"Detailed metrics saved to {metrics_path}")

        return {
            'y_true': y_true, 'y_pred': y_pred, 'class_names': class_names,
            'report': report, 'confusion_matrix': conf_matrix, 'accuracy': accuracy
        }

if __name__ == "__main__":
    classifier = EmotionClassifier()
    classifier.load_data()
    classifier.print_class_distribution() # Good to check loaded data
    classifier.build_model()
    # classifier.model.summary() # Useful for debugging model structure

    history = classifier.train()
    # print("\nTraining History:", history) # For debugging training process
    
    classifier.evaluate()