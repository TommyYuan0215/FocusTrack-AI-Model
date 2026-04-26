import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.applications.resnet50 import preprocess_input  # Use for ResNet50
import mediapipe as mp
from config.config import Config


class RealtimeInference:
    """Real-time webcam inference using a trained emotion recognition model."""

    def __init__(self):
        self.class_names = Config.CLASS_NAMES
        self.model = self._load_model()
        self.mp_face_detection = mp.solutions.face_detection

    @staticmethod
    def _load_model():
        """Load the trained model from disk."""
        model_path = os.path.join(Config.MODEL_DIR, Config.MODEL_FILENAME)
        try:
            model = load_model(model_path, compile=False)
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise SystemExit(1)

    def _predict_face(self, frame, x, y, w, h):
        """Crop, preprocess, and classify a detected face region."""
        ih, iw, _ = frame.shape

        # Clamp coordinates to frame boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(iw, x + w)
        y2 = min(ih, y + h)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None, None

        face_resized = cv2.resize(face, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(np.expand_dims(face_rgb.astype(np.float32), axis=0))

        prediction = self.model.predict(face_input, verbose=0)
        predicted_class = int(np.argmax(prediction, axis=-1)[0])
        confidence = float(prediction[0][predicted_class])

        label = self.class_names.get(predicted_class, "Unknown")
        return label, confidence

    def run(self, camera_index=0, frame_width=640, frame_height=480):
        """Start real-time webcam inference loop. Press 'q' to quit."""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        if not cap.isOpened():
            print("Error: Cannot open camera.")
            return

        print("Starting real-time inference. Press 'q' to quit.")

        with self.mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x = int(bboxC.xmin * iw)
                        y = int(bboxC.ymin * ih)
                        w = int(bboxC.width * iw)
                        h = int(bboxC.height * ih)

                        label, confidence = self._predict_face(frame, x, y, w, h)

                        if label is not None:
                            # Draw bounding box and label
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            display_text = f"{label} ({confidence:.2f})"
                            cv2.putText(
                                frame, display_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                            )

                cv2.imshow('Real-Time Emotion Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self._cleanup(cap)

    @staticmethod
    def _cleanup(cap):
        """Release camera and close windows."""
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    inference = RealtimeInference()
    inference.run()