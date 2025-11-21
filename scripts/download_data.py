import os
import requests
import tarfile
from tqdm import tqdm

URL = "https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=1"
DEST_DIR = "data/raw"
ARCHIVE_NAME = "yearbook.tar.gz"
ARCHIVE_PATH = os.path.join(DEST_DIR, ARCHIVE_NAME)

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)
    print("Download complete.")

def extract_file(archive_path, dest_dir):
    print(f"Extracting {archive_path} to {dest_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)
    print("Extraction complete.")

if __name__ == "__main__":
    os.makedirs(DEST_DIR, exist_ok=True)
    if not os.path.exists(ARCHIVE_PATH):
        download_file(URL, ARCHIVE_PATH)
    else:
        print(f"{ARCHIVE_PATH} already exists. Skipping download.")
    
    extract_file(ARCHIVE_PATH, DEST_DIR)
