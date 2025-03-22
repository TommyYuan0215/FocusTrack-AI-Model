import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import multiprocessing as mp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config.config import Config

class VideoPreprocessor:
    def __init__(self, use_fast_detector=True):
        self.dataset_dir = Config.DATASET_DIR
        self.output_base_dir = Config.PROCESSED_DATA_DIR
        self.emotion_columns = Config.EMOTION_COLUMNS
        self.target_size = (224, 224)  # VGGFace expected input size
        self.use_fast_detector = use_fast_detector
        
        # Get the full path of the cascade file for later use
        self.cascade_path = Config.FACE_CASCADE_PATH
        # Verify that the cascade file exists
        if not os.path.exists(self.cascade_path):
            raise FileNotFoundError(f"Cascade file not found at {self.cascade_path}")
        
        # VGGFace-appropriate augmentation
        self.augmenter = ImageDataGenerator(
            rotation_range=2,  # Reduce from 3
            width_shift_range=0.01,  # Reduce from 0.02
            height_shift_range=0.01,  # Reduce from 0.02
            zoom_range=0.01,  # Reduce from 0.02
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    def vggface_preprocess(self, img):
        """Apply VGGFace-specific preprocessing"""
        if img.shape[-1] == 3:
            img = img[..., ::-1]  # Convert BGR to RGB
        
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863
        return img
    
    def get_dominant_label(self, row, confidence_threshold=0.5):
        """Determine the dominant emotion label with confidence threshold."""
        row = row.rename(str.strip)
        scores = row[self.emotion_columns]
        
        if scores.sum() == 0:
            return None
            
        max_score = scores.max()
        if max_score < confidence_threshold:
            return None
            
        max_labels = scores[scores == max_score].index.tolist()
        return random.choice(max_labels).strip().lower() if max_labels else None

    @staticmethod
    def detect_and_align_face_fast(image, cascade_path, target_size=(224, 224)):
        """Static method for fast face detection using OpenCV's Haar Cascade."""
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print(f"Error: Failed to load cascade classifier from {cascade_path}")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None

        # Get the largest face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        margin_x, margin_y = int(w * 0.1), int(h * 0.1)
        x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
        x2, y2 = min(image.shape[1], x + w + margin_x), min(image.shape[0], y + h + margin_y)

        face_img = image[y1:y2, x1:x2]

        if face_img.size == 0:
            return None
        
        return cv2.resize(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), target_size, interpolation=cv2.INTER_LANCZOS4)
    
    @staticmethod
    def detect_and_align_face_mtcnn(image, target_size=(224, 224)):
        """Static method for MTCNN-based face detection and alignment."""
        if image is None:
            return None
            
        # Only import MTCNN here when needed
        try:
            from mtcnn import MTCNN
            face_detector = MTCNN()
        except ImportError:
            print("MTCNN not available. Please install it with 'pip install mtcnn'")
            return None
            
        # Convert to RGB for MTCNN
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return None
            
        # Detect faces
        faces = face_detector.detect_faces(rgb_image)
        if not faces:
            return None
            
        # Get the face with highest confidence
        face = max(faces, key=lambda x: x['confidence'])
        if face['confidence'] < 0.95:  # Filter low confidence detections
            return None
            
        # Extract face coordinates
        x, y, w, h = face['box']
        
        # Get facial landmarks for alignment
        left_eye = face['keypoints']['left_eye']
        right_eye = face['keypoints']['right_eye']
        
        # Calculate angle for alignment
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate to align
        center_x = (left_eye[0] + right_eye[0]) // 2
        center_y = (left_eye[1] + right_eye[1]) // 2
        center = (int(center_x), int(center_y))
        
        M = cv2.getRotationMatrix2D(center, angle, 1)
        aligned_image = cv2.warpAffine(rgb_image, M, (rgb_image.shape[1], rgb_image.shape[0]))
        
        # Extract aligned face with margin
        margin_x, margin_y = int(w * 0.3), int(h * 0.3)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(aligned_image.shape[1], x + w + margin_x)
        y2 = min(aligned_image.shape[0], y + h + margin_y)
        
        face_img = aligned_image[y1:y2, x1:x2]
        
        # Check face quality
        if face_img.size == 0:
            return None
        
        # Resize to VGGFace input size
        return cv2.resize(face_img, target_size, interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def extract_frames(video_path, output_dir, cascade_path, augmenter=None, 
                    use_fast_detector=True, interval=1.0, max_frames=15, target_size=(224, 224)):
        """Static method to extract frames from the video and save detected faces."""
        
        os.makedirs(output_dir, exist_ok=True)
        vidcap = cv2.VideoCapture(video_path)

        if not vidcap.isOpened():
            print(f"Error: Cannot open video file: {video_path}")
            return 0

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_interval = max(1, int(fps * interval))
        frames_to_extract = [i for i in range(0, total_frames, frame_interval)][:max_frames]

        frames_extracted = 0 

        for frame_pos in frames_to_extract:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            success, image = vidcap.read()

            if not success:
                print(f"Warning: Could not read frame {frame_pos}, skipping.")
                continue

            # Detect face
            if use_fast_detector:
                face_img = VideoPreprocessor.detect_and_align_face_fast(image, cascade_path, target_size)
            else:
                face_img = VideoPreprocessor.detect_and_align_face_mtcnn(image, target_size)

            # Skip if no face is detected
            if face_img is None or face_img.size == 0:
                print(f"No valid face detected in frame {frame_pos}, skipping.")
                continue

            # Resize if necessary
            if face_img.shape[:2] != target_size:
                face_img = cv2.resize(face_img, target_size, interpolation=cv2.INTER_AREA)

            # Normalize image
            face_img_resnet = face_img.astype(np.float32) / 255.0  

            # Convert to uint8 for saving
            face_img_save = (face_img_resnet * 255).clip(0, 255).astype(np.uint8)

            # Save the processed frame
            frame_filename = os.path.join(output_dir, f"frame_{frames_extracted:03d}.jpg")
            savedFrame_success = cv2.imwrite(frame_filename, cv2.cvtColor(face_img_save, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

            if savedFrame_success:
                frames_extracted += 1 
            else:
                print(f"Error: Could not save frame {frame_pos}")

        vidcap.release()

        return frames_extracted


    @staticmethod
    def process_video_worker(args):
        """Static method for processing a single video (for parallel processing)."""
        row, split, dataset_dir, output_base_dir, emotion_columns, use_fast_detector, target_size, cascade_path = args
        
        try:
            temp_processor = VideoPreprocessor(use_fast_detector=use_fast_detector)
            temp_processor.emotion_columns = emotion_columns

            clip_id = row['ClipID']
            label = temp_processor.get_dominant_label(row, confidence_threshold=0.6)

            if label is None:
                return None

            video_path = os.path.join(dataset_dir, split, clip_id[:6], clip_id[:-4], clip_id)
            output_folder = os.path.join(output_base_dir, split, label, clip_id[:-4])

            if not os.path.exists(video_path):
                return None

            frames_count = VideoPreprocessor.extract_frames(
                video_path, output_folder, cascade_path, augmenter=temp_processor.augmenter,
                use_fast_detector=use_fast_detector, target_size=target_size
            )

            if frames_count > 0:
                return {'clip_id': clip_id, 'emotion': label, 'split': split, 'frames': frames_count, 'path': output_folder}
        
        except Exception as e:
            print(f"Error processing {row['ClipID']}: {e}")
            return None

    # def balance_dataset(self, data_df, target_samples_per_class=None):
    #     """Balances the dataset with a mix of downsampling and oversampling."""
        
    #     # Get class distribution
    #     emotions = data_df['emotion'].tolist()
    #     class_counts = Counter(emotions)
        
    #     # Set target sample size per class
    #     if target_samples_per_class is None:
    #         target_samples_per_class = int(np.percentile(list(class_counts.values()), 50)) 

    #     balanced_df = pd.DataFrame()

    #     for emotion, count in class_counts.items():
    #         emotion_df = data_df[data_df['emotion'] == emotion]

    #         # Downsample if the class is too large
    #         if count > target_samples_per_class:
    #             emotion_df = emotion_df.sample(target_samples_per_class, random_state=42)

    #         balanced_df = pd.concat([balanced_df, emotion_df])

    #     # Create a dictionary for sampling strategy
    #     # This maps each class to its target sample count (capped at 500)
    #     class_distribution = Counter(balanced_df['emotion'])
    #     sampling_strategy = {emotion: min(500, max(class_distribution.values())) 
    #                         for emotion in class_distribution.keys()}
        
    #     # Use the dictionary as sampling_strategy
    #     ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
    #     balanced_df, _ = ros.fit_resample(balanced_df, balanced_df['emotion'])

    #     return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    def process_dataset(self, num_workers=None):
        """Process videos in parallel with improved performance."""
        if num_workers is None:
            # Use all available cores except one
            num_workers = max(1, mp.cpu_count() - 1)
            
        print(f"Processing with {num_workers} workers...")
        
        # Verify the cascade file exists before starting processing
        if self.use_fast_detector and not os.path.exists(self.cascade_path):
            raise FileNotFoundError(f"Cascade file not found at {self.cascade_path}")
            
        # Create metadata list to track processed data
        metadata_list = []
        
        for split, csv_file in Config.LABEL_FILES.items():
            print(f"Processing {split} split...")
            df = pd.read_csv(csv_file)
            
            # Prepare arguments for parallel processing
            # Include all necessary data as args to avoid pickling the class instance
            args_list = [
                (row, split, self.dataset_dir, self.output_base_dir, 
                 self.emotion_columns, self.use_fast_detector, self.target_size, 
                 self.cascade_path) 
                for _, row in df.iterrows()
            ]
            
            # Process videos in parallel
            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(pool.imap(VideoPreprocessor.process_video_worker, args_list), total=len(args_list)))
            
            # Filter out None results and add to metadata
            valid_results = [result for result in results if result is not None]
            metadata_list.extend(valid_results)
        
        # Convert to DataFrame
        metadata_df = pd.DataFrame(metadata_list) if metadata_list else pd.DataFrame()
        
        # Save metadata if we have results
        if not metadata_df.empty:
            metadata_df.to_csv(os.path.join(self.output_base_dir, 'metadata.csv'), index=False)
            
            # # Balance dataset within each split independently
            # print("Balancing dataset...")
            # balanced_df = pd.DataFrame()
            
            # for split in metadata_df['split'].unique():
            #     split_df = metadata_df[metadata_df['split'] == split]
            #     balanced_split_df = self.balance_dataset(split_df)
            #     balanced_df = pd.concat([balanced_df, balanced_split_df])
            
            # balanced_df.to_csv(os.path.join(self.output_base_dir, 'balanced_metadata.csv'), index=False)
            
            # Print statistics
            print("\nDataset statistics:")
            for split in metadata_df['split'].unique():
                split_df = metadata_df[metadata_df['split'] == split]
                print(f"\n{split.upper()} split:")
                print(f"Total clips: {len(split_df)}")
                print("Emotion distribution:")
                for emotion, count in Counter(split_df['emotion']).items():
                    print(f"  {emotion}: {count}")
        else:
            print("No videos were successfully processed.")

if __name__ == "__main__":
    # Limit TensorFlow memory usage
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass
    
    # Create the preprocessor with fast detector
    preprocessor = VideoPreprocessor(use_fast_detector=True)
    
    # Process the dataset with parallel processing
    preprocessor.process_dataset()