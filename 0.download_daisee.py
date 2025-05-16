# download_dataset.py
import os
import requests
import zipfile
import shutil
from tqdm import tqdm
from config.config import Config

def download_and_extract_daisee():
    """Download DAiSEE dataset and extract only Dataset and Label folders to raw directory"""
    url = "https://download.link/to/daisee_dataset.zip"  # Replace with actual URL
    target_dir = Config.RAW_DIR
    temp_extract_dir = "temp_daisee_extract"
    zip_path = "daisee_temp.zip"
    
    # Create directories if they don't exist
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    # Download file with progress bar
    print(f"Downloading DAiSEE dataset...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    
    # Extract the zip file to temp directory
    print("Extracting files to temporary directory...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
    
    # Move only Dataset and Label folders to target directory
    print("Moving Dataset and Label folders to raw directory...")
    
    # Find Dataset folder (might be nested)
    dataset_source = None
    label_source = None
    
    for root, dirs, files in os.walk(temp_extract_dir):
        for dir_name in dirs:
            if dir_name == "Dataset":
                dataset_source = os.path.join(root, dir_name)
            elif dir_name == "Label":
                label_source = os.path.join(root, dir_name)
    
    # Move the folders if found
    if dataset_source:
        dataset_dest = os.path.join(target_dir, "DataSet")
        if os.path.exists(dataset_dest):
            shutil.rmtree(dataset_dest)
        shutil.move(dataset_source, dataset_dest)
        print(f"Moved Dataset to {dataset_dest}")
    else:
        print("Dataset folder not found in the extracted files")
    
    if label_source:
        label_dest = os.path.join(target_dir, "Label")
        if os.path.exists(label_dest):
            shutil.rmtree(label_dest)
        shutil.move(label_source, label_dest)
        print(f"Moved Label to {label_dest}")
    else:
        print("Label folder not found in the extracted files")
    
    # Clean up
    os.remove(zip_path)
    shutil.rmtree(temp_extract_dir)
    print("Temporary files cleaned up")
    print(f"Extraction complete - Dataset and Label folders are in {target_dir}")

if __name__ == "__main__":
    download_and_extract_daisee()