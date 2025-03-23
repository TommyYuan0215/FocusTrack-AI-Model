import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import mediapipe as mp
from config.config import Config

# Load the pre-trained model
try:
    model = load_model(os.path.join(Config.MODEL_DIR, 'emotion_recognition_model_test.h5'))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Start video capture (0 is for the default camera)
cap = cv2.VideoCapture(0)

# Adjust camera resolution if necessary
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe Face Detection model
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to RGB (MediaPipe uses RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect faces
        results = face_detection.process(rgb_frame)

        # If faces are detected
        if results.detections:
            for detection in results.detections:
                # Get bounding box coordinates of the face
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                              int(bboxC.width * iw), int(bboxC.height * ih)

                # Draw the bounding box around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Crop the detected face from the frame
                face = frame[y:y + h, x:x + w]

                # Resize the face to the size required by your model (typically 224x224 for VGG)
                face_resized = cv2.resize(face, (224, 224))

                # Convert the image to RGB (OpenCV loads in BGR format)
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                # Expand dimensions to match the model's input shape (batch size, height, width, channels)
                face_resized = np.expand_dims(face_resized, axis=0)

                # Preprocess the image for the model (this depends on the model used)
                face_resized = preprocess_input(face_resized)

                # Make a prediction with the pre-trained model
                prediction = model.predict(face_resized)

                # For multi-class predictions, get the index of the maximum probability
                predicted_class = np.argmax(prediction, axis=-1)

                # Map the predicted class to a label (you can customize this part)
                labels = ['Boredom', 'Engagement', 'Frustration', 'Confusion']  # Modify based on your model
                predicted_label = labels[predicted_class[0]]

                # Display the predicted label above the face
                cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with detected faces and predictions
        cv2.imshow('Real-Time Face Detection and Prediction', frame)

        # Exit the loop when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()