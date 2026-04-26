import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import multiprocessing as mp
from config.config import Config


class VideoPreprocessor:
    """Preprocesses DAiSEE video data: extracts frames, detects faces, and saves cropped face images."""

    def __init__(self):
        self.dataset_dir = Config.DATASET_DIR
        self.output_base_dir = Config.PROCESSED_DATA_DIR
        self.emotion_columns = Config.EMOTION_COLUMNS
        self.target_size = (224, 224)

        self.cascade_path = Config.FACE_CASCADE_PATH
        if not os.path.exists(self.cascade_path):
            raise FileNotFoundError(f"Cascade file not found at {self.cascade_path}")

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
    def detect_and_crop_face(image, cascade_path, target_size=(224, 224)):
        """Detect the largest face in an image using Haar Cascade and return a cropped, resized face."""
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print(f"Error: Failed to load cascade classifier from {cascade_path}")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return None

        # Select the largest detected face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        # Add a small margin around the face
        margin_x, margin_y = int(w * 0.1), int(h * 0.1)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)

        face_img = image[y1:y2, x1:x2]
        if face_img.size == 0:
            return None

        # Convert BGR→RGB and resize to model input dimensions
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        return cv2.resize(face_rgb, target_size, interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def extract_frames(video_path, output_dir, cascade_path,
                       interval=1.0, max_frames=15, target_size=(224, 224)):
        """Extract frames from a video at fixed intervals, detect faces, and save cropped face images."""
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
                continue

            face_img = VideoPreprocessor.detect_and_crop_face(image, cascade_path, target_size)
            if face_img is None:
                continue

            # Save as high-quality JPEG (face_img is RGB, convert back to BGR for imwrite)
            frame_filename = os.path.join(output_dir, f"frame_{frames_extracted:03d}.jpg")
            saved = cv2.imwrite(
                frame_filename,
                cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            if saved:
                frames_extracted += 1
            else:
                print(f"Error: Could not save frame {frame_pos}")

        vidcap.release()
        return frames_extracted

    @staticmethod
    def process_video_worker(args):
        """Worker function for parallel video processing.

        Receives a tuple of (row_dict, split, config_values) to avoid
        creating a full VideoPreprocessor instance per video.
        """
        row_dict, split, config_values = args
        dataset_dir, output_base_dir, cascade_path, target_size, emotion_columns = config_values

        try:
            row = pd.Series(row_dict)
            clip_id = row['ClipID']

            # Determine dominant label
            row_renamed = row.rename(str.strip)
            scores = row_renamed[emotion_columns]
            if scores.sum() == 0:
                return None
            max_score = scores.max()
            if max_score < 0.6:  # confidence threshold
                return None
            max_labels = scores[scores == max_score].index.tolist()
            label = random.choice(max_labels).strip().lower() if max_labels else None
            if label is None:
                return None

            video_path = os.path.join(dataset_dir, split, clip_id[:6], clip_id[:-4], clip_id)
            output_folder = os.path.join(output_base_dir, split, label, clip_id[:-4])

            if not os.path.exists(video_path):
                return None

            frames_count = VideoPreprocessor.extract_frames(
                video_path, output_folder, cascade_path, target_size=target_size
            )

            if frames_count > 0:
                return {
                    'clip_id': clip_id,
                    'emotion': label,
                    'split': split,
                    'frames': frames_count,
                    'path': output_folder,
                }

        except Exception as e:
            print(f"Error processing {row_dict.get('ClipID', 'unknown')}: {e}")

        return None

    def process_dataset(self, num_workers=None):
        """Process all video splits in parallel, extract face frames, and save metadata."""
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        print(f"Processing with {num_workers} workers...")

        if not os.path.exists(self.cascade_path):
            raise FileNotFoundError(f"Cascade file not found at {self.cascade_path}")

        # Pack config values once — passed to every worker to avoid per-video instance creation
        config_values = (
            self.dataset_dir,
            self.output_base_dir,
            self.cascade_path,
            self.target_size,
            self.emotion_columns,
        )

        metadata_list = []

        for split, csv_file in Config.LABEL_FILES.items():
            print(f"Processing {split} split...")
            df = pd.read_csv(csv_file)

            # Convert rows to dicts so they are picklable for multiprocessing
            args_list = [(row.to_dict(), split, config_values) for _, row in df.iterrows()]

            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(pool.imap(VideoPreprocessor.process_video_worker, args_list), total=len(args_list)))

            valid_results = [r for r in results if r is not None]
            metadata_list.extend(valid_results)

        metadata_df = pd.DataFrame(metadata_list) if metadata_list else pd.DataFrame()

        if not metadata_df.empty:
            metadata_df.to_csv(os.path.join(self.output_base_dir, 'metadata.csv'), index=False)

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
    preprocessor = VideoPreprocessor()
    preprocessor.process_dataset()