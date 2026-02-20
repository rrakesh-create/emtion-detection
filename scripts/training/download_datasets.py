import os
import requests
import zipfile
import io
from tqdm import tqdm

def download_file(url, target_path):
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(target_path, 'wb') as f, tqdm(
        desc=target_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def setup_ravdess(base_dir):
    ravdess_dir = os.path.join(base_dir, "Ravdess")
    if not os.path.exists(ravdess_dir):
        os.makedirs(ravdess_dir)
    
    # RAVDESS Audio Speech (approx 200MB)
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    zip_path = os.path.join(ravdess_dir, "ravdess_audio.zip")
    
    if not os.path.exists(zip_path) and not os.path.exists(os.path.join(ravdess_dir, "Actor_01")):
        download_file(url, zip_path)
        extract_zip(zip_path, ravdess_dir)
        # Cleanup zip
        # os.remove(zip_path) 
        print("RAVDESS setup complete.")
    else:
        print("RAVDESS already present.")

def setup_tess(base_dir):
    # TESS is harder to direct download without Kaggle API. 
    # We will skip auto-download for now or use a backup source if found.
    print("TESS download requires manual steps or Kaggle API. Skipping for now.")
    pass

if __name__ == "__main__":
    # Base datasets dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # mers/train -> mers -> datasets
    project_root = os.path.dirname(os.path.dirname(current_dir))
    datasets_dir = os.path.join(project_root, "datasets")
    
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
        
    print(f"Setting up datasets in {datasets_dir}...")
    setup_ravdess(datasets_dir)
