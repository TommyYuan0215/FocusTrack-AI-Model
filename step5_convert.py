import os
import tensorflow as tf
from config.config import Config

class ModelConverter:
    """Converts a trained Keras/H5 model to an optimized TensorFlow Lite model."""

    def __init__(self):
        self.model_path = os.path.join(Config.MODEL_DIR, Config.MODEL_FILENAME)
        
        # Ensure the new file has a .tflite extension regardless of the original format
        self.tflite_model_path = os.path.join(Config.MODEL_DIR, Config.TFLITE_MODEL_FILENAME)

    def convert(self):
        print(f"Loading the trained model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            print("Error: Trained model not found. Run step4_train.py first.")
            return

        try:
            # 1. Load the existing model
            model = tf.keras.models.load_model(self.model_path, compile=False)

            # 2. Initialize the TFLite Converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # 3. Apply default optimizations (Shrinks size and speeds up CPU inference)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            print("Converting to TFLite format (this may take a minute)...")
            
            # 4. Convert the model
            tflite_model = converter.convert()

            # 5. Save the new .tflite file
            with open(self.tflite_model_path, "wb") as f:
                f.write(tflite_model)

            print(f"Success! TFLite model saved to: {self.tflite_model_path}")
            
        except Exception as e:
            print(f"Error during TFLite conversion: {e}")

if __name__ == "__main__":
    converter = ModelConverter()
    converter.convert()