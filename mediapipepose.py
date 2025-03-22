import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and Face Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize MediaPipe drawing utils to visualize pose landmarks and face detection
mp_drawing = mp.solutions.drawing_utils

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

def calculate_engagement_score(landmarks, face_detected):
    # Define key landmarks for engagement detection
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]

    # Check if the left hand is raised (left elbow is above the left shoulder)
    left_hand_raised = left_elbow.y < left_shoulder.y

    # Check if the right hand is raised (right elbow is above the right shoulder)
    right_hand_raised = right_elbow.y < right_shoulder.y

    # If no face detected, return 0 (not engaged)
    if not face_detected:
        return 0.0

    # Calculate engagement score based on the raised hands
    if left_hand_raised or right_hand_raised:
        return 1.0  
    else:
        return 0.8  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get face detection results
    # face_results = face_detection.process(rgb_frame)
    
    # Process the frame to get pose landmarks
    pose_results = pose.process(rgb_frame)

    # Check if face is detected (face_detection.results will be None if no face detected)
    # face_detected = face_results.detections is not None

    # Check if pose landmarks are detected
    if pose_results.pose_landmarks:
        # Draw pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks for engagement check
        landmarks = pose_results.pose_landmarks.landmark
        engagement_score = calculate_engagement_score(landmarks, pose_results)

        # Display engagement score on the frame
        cv2.putText(frame, f'Engagement Score: {engagement_score:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame with labels
    cv2.imshow('MediaPipe Pose with Face Detection and Engagement Score', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
