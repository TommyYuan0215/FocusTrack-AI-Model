import os
import numpy as np
import pandas as pd
import shutil
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from config.config import Config


class BalancedDataGenerator:
    """Balances the preprocessed dataset by downsampling large classes and oversampling small ones."""

    def __init__(self, target_train_samples=None):
        self.processed_data_dir = Config.PROCESSED_DATA_DIR
        self.backup_dir = Config.ORIGINAL_PROCESSED_DATA_DIR
        self.input_metadata = Config.INPUT_METADATA
        self.output_metadata = Config.OUTPUT_METADATA
        self.target_train_samples = target_train_samples

    def backup_original_data(self):
        """Backup the original dataset before modifying it."""
        if os.path.exists(self.backup_dir):
            print("Backup already exists. Skipping...")
            return

        os.makedirs(self.backup_dir, exist_ok=True)
        shutil.copy2(self.input_metadata, os.path.join(self.backup_dir, 'metadata.csv'))

        data_df = pd.read_csv(self.input_metadata)
        for split in data_df['split'].unique():
            for emotion in data_df[data_df['split'] == split]['emotion'].unique():
                src_dir = os.path.join(self.processed_data_dir, split, str(emotion))
                dst_dir = os.path.join(self.backup_dir, split, str(emotion))
                if os.path.exists(src_dir):
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        print("Backup completed.")

    def balance_dataset(self):
        """Balance the training set while preserving validation and test sets."""
        if not os.path.exists(self.processed_data_dir) or not os.path.exists(self.input_metadata):
            print("Error: Required data or metadata is missing.")
            return

        if os.path.exists(self.output_metadata):
            print("Balanced metadata already exists. Skipping...")
            return

        self.backup_original_data()

        data_df = pd.read_csv(self.input_metadata)
        train_df = data_df[data_df['split'] == 'train']
        val_df = data_df[data_df['split'] == 'val']
        test_df = data_df[data_df['split'] == 'test']

        print("\nBefore Balancing (Training Set):")
        print(train_df['emotion'].value_counts())

        # Determine target samples per class
        class_counts = Counter(train_df['emotion'])
        if self.target_train_samples is None:
            self.target_train_samples = max(int(np.mean(list(class_counts.values())) * 1.2), 5000)

        # Downsample over-represented classes
        balanced_train_df = pd.DataFrame()
        for emotion, count in class_counts.items():
            emotion_df = train_df[train_df['emotion'] == emotion]
            if count > self.target_train_samples:
                emotion_df = emotion_df.sample(self.target_train_samples, random_state=42)
            balanced_train_df = pd.concat([balanced_train_df, emotion_df])

        # Oversample under-represented classes
        ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
        balanced_train_df, _ = ros.fit_resample(balanced_train_df, balanced_train_df['emotion'])

        # Verify all referenced paths exist
        missing_paths = [path for path in balanced_train_df['path'] if not os.path.exists(path)]
        if missing_paths:
            print(f"WARNING: {len(missing_paths)} missing files in the training set.")

        print("\nAfter Balancing (Training Set):")
        print(balanced_train_df['emotion'].value_counts())

        # Combine balanced train with untouched val/test and save
        final_df = pd.concat([balanced_train_df, val_df, test_df])
        final_df.to_csv(self.output_metadata, index=False)
        os.remove(self.input_metadata)

        print("Balancing complete. Metadata updated.")


if __name__ == "__main__":
    generator = BalancedDataGenerator(target_train_samples=5000)
    generator.balance_dataset()
