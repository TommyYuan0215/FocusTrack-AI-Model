import os
import pandas as pd
import numpy as np
import shutil
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from config.config import Config

class DatasetBalancer:
    def __init__(self, target_samples_per_class=None):
        self.processed_data_dir = Config.PROCESSED_DATA_DIR
        self.original_processed_backup_dir = os.path.join(Config.DATA_DIR, 'original_processed')
        self.input_metadata = os.path.join(self.processed_data_dir, 'metadata.csv')
        self.output_metadata = os.path.join(self.processed_data_dir, 'balanced_metadata.csv')
        self.target_samples_per_class = target_samples_per_class
    
    def backup_original_data(self):
        """Create a backup of the original dataset"""
        print("Creating backup of original dataset...")
        
        # Don't overwrite existing backup
        if os.path.exists(self.original_processed_backup_dir):
            print(f"Backup already exists at {self.original_processed_backup_dir}")
            return
            
        # Copy the metadata file
        os.makedirs(self.original_processed_backup_dir, exist_ok=True)
        shutil.copy2(self.input_metadata, os.path.join(self.original_processed_backup_dir, 'metadata.csv'))
        
        # Read the metadata to know what to copy
        data_df = pd.read_csv(self.input_metadata)
        
        # Copy all the data folders
        for split in data_df['split'].unique():
            for emotion in data_df[data_df['split'] == split]['emotion'].unique():
                src_dir = os.path.join(self.processed_data_dir, split, emotion)
                dst_dir = os.path.join(self.original_processed_backup_dir, split, emotion)
                
                if os.path.exists(src_dir):
                    print(f"Backing up {src_dir} to {dst_dir}")
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        
        print(f"Original dataset backed up to {self.original_processed_backup_dir}")
    
    def balance_dataset(self):
        """Balances the dataset by modifying the original processed data directory."""
        
        if not os.path.exists(self.processed_data_dir):
            print("Error: Processed data directory does not exist! Exiting...")
            return
        
        if not os.path.exists(self.input_metadata):
            print("Error: metadata.csv not found in the processed directory! Exiting...")
            return

        if os.path.exists(self.output_metadata):
            print("Balanced metadata already exists! Skipping processing...")
            return
        
        # Check original data is exists or not
        if os.path.exists(self.original_processed_backup_dir):
            print("Notice: Backup data already exists, passing to the next step....")
        else:
            self.backup_original_data()
        
        print("Loading metadata...")
        data_df = pd.read_csv(self.input_metadata)
        
        # Get class distribution
        class_counts = Counter(data_df['emotion'])
        print("Original class distribution:")
        for emotion, count in class_counts.items():
            print(f"  {emotion}: {count}")
        
        # Set a reasonable target sample size per class
        if self.target_samples_per_class is None:
            self.target_samples_per_class = max(int(np.mean(list(class_counts.values())) * 1.2), 500)
        
        # Get sample list for each emotion
        balanced_df = pd.DataFrame()
        
        # Perform downsampling
        for emotion, count in class_counts.items():
            emotion_df = data_df[data_df['emotion'] == emotion]
            
            # Gradual downsampling (not too aggressive)
            if count > self.target_samples_per_class:
                downsample_size = max(int(count * 0.5), self.target_samples_per_class)  # Reduce by max 50%
                emotion_df = emotion_df.sample(downsample_size, random_state=42)
                
                # Get samples to remove
                samples_to_remove = set(data_df[data_df['emotion'] == emotion]['path']) - set(emotion_df['path'])
                
                # Remove these directories
                for path in samples_to_remove:
                    if os.path.exists(path):
                        print(f"Removing: {path}")
                        shutil.rmtree(path)
            
            balanced_df = pd.concat([balanced_df, emotion_df])
        
        # Create a dictionary for oversampling strategy
        class_distribution = Counter(balanced_df['emotion'])
        max_class_size = max(class_distribution.values())
        sampling_strategy = {
            emotion: max(count, max_class_size)  # Oversample only if needed
            for emotion, count in class_distribution.items()
        }
        
        # Apply oversampling
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        balanced_df, _ = ros.fit_resample(balanced_df, balanced_df['emotion'])
        
        # Find samples that need to be duplicated (created by oversampling)
        original_paths = set(data_df['path'])
        oversampled_df = balanced_df[~balanced_df['path'].isin(original_paths)]
        
        # Duplicate these samples
        print("\nDuplicating samples for oversampling...")
        for idx, row in oversampled_df.iterrows():
            # Find a source to copy from (any existing sample of the same emotion)
            source_sample = data_df[data_df['emotion'] == row['emotion']].iloc[0]
            source_path = source_sample['path']
            
            # Create a new unique path 
            split = row['split']
            emotion = row['emotion']
            clip_id = row['clip_id']
            clip_id_without_ext = clip_id[:-4]
            
            # Add a unique suffix to avoid collisions
            new_id = f"{clip_id_without_ext}_dup{idx}"
            target_path = os.path.join(self.processed_data_dir, split, emotion, new_id)
            
            # Ensure the directory structure exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Copy the files
            if os.path.exists(source_path) and not os.path.exists(target_path):
                try:
                    shutil.copytree(source_path, target_path)
                    print(f"Duplicated: {source_path} -> {target_path}")
                    # Update the path in the dataframe
                    balanced_df.loc[idx, 'path'] = target_path
                except Exception as e:
                    print(f"Error duplicating {source_path}: {e}")
        
        # Save the balanced metadata
        print("Saving balanced metadata...")
        balanced_df.to_csv(self.output_metadata, index=False)
        
        # Remove original metadata in current processed directory
        os.remove(self.input_metadata)
        
        # Print statistics
        print("\nBalanced dataset statistics:")
        for split in balanced_df['split'].unique():
            split_df = balanced_df[balanced_df['split'] == split]
            print(f"\n{split.upper()} split:")
            print(f"Total clips: {len(split_df)}")
            print("Emotion distribution:")
            for emotion, count in Counter(split_df['emotion']).items():
                print(f"  {emotion}: {count}")
        
        print("\nDataset balancing completed!")
        print(f"Original dataset backed up to: {self.original_processed_backup_dir}")
        print(f"Balanced dataset is now in: {self.processed_data_dir}")
        print(f"Balanced metadata saved to: {self.output_metadata}")
        
if __name__ == "__main__":   
    balancer = DatasetBalancer()
    balancer.balance_dataset()