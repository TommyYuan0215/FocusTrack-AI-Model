import os
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from config.config import Config

class DatasetBalancer:
    def __init__(self, target_samples_per_class=None):
        self.input_metadata = os.path.join(Config.PROCESSED_DATA_DIR, 'metadata.csv')
        self.output_metadata = os.path.join(Config.PROCESSED_DATA_DIR, 'balanced_metadata.csv')
        self.target_samples_per_class = target_samples_per_class
    
    def balance_dataset(self):
        """Balances the dataset using controlled downsampling and oversampling."""
        print("Loading metadata...")
        data_df = pd.read_csv(self.input_metadata)
        
        # Get class distribution
        class_counts = Counter(data_df['emotion'])
        
        # Set a reasonable target sample size per class
        if self.target_samples_per_class is None:
            self.target_samples_per_class = max(
                int(np.percentile(list(class_counts.values()), 75)), 500
            )  # 75th percentile or at least 500
        
        balanced_df = pd.DataFrame()

        for emotion, count in class_counts.items():
            emotion_df = data_df[data_df['emotion'] == emotion]
            
            # Gradual downsampling (not too aggressive)
            if count > self.target_samples_per_class:
                downsample_size = max(int(count * 0.5), self.target_samples_per_class)  # Reduce by max 50%
                emotion_df = emotion_df.sample(downsample_size, random_state=42)
            
            balanced_df = pd.concat([balanced_df, emotion_df])
        
        # Create a dictionary for oversampling strategy
        class_distribution = Counter(balanced_df['emotion'])
        max_class_size = max(class_distribution.values())
        sampling_strategy = {emotion: min(1000, max_class_size) for emotion in class_distribution.keys()}
        
        # Apply oversampling
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        balanced_df, _ = ros.fit_resample(balanced_df, balanced_df['emotion'])

        # Shuffle and reset index
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print("Saving balanced dataset...")
        balanced_df.to_csv(self.output_metadata, index=False)
        
        # Print statistics
        print("\nDataset statistics:")
        for split in balanced_df['split'].unique():
            split_df = balanced_df[balanced_df['split'] == split]
            print(f"\n{split.upper()} split:")
            print(f"Total clips: {len(split_df)}")
            print("Emotion distribution:")
            for emotion, count in Counter(split_df['emotion']).items():
                print(f"  {emotion}: {count}")
        
        print("Balanced dataset saved successfully!")
        
if __name__ == "__main__":   
    balancer = DatasetBalancer()
    balancer.balance_dataset()