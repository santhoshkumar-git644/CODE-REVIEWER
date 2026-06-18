import os
import sys
import urllib.request
import zipfile
import tarfile
from typing import List

def download_file(url: str, dest_path: str):
    """Downloads a file with progress tracking."""
    if os.path.exists(dest_path):
        print(f"File {dest_path} already exists. Skipping download.")
        return

    print(f"Downloading {url} to {dest_path}...")
    try:
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 1e2 / totalsize
                s = f"\r{percent:5.1f}% {readsofar} / {totalsize}"
                sys.stdout.write(s)
                if readsofar >= totalsize: # near the end
                    sys.stdout.write("\n")
            else: # total size is unknown
                sys.stdout.write(f"\rread {readsofar}")
                
        urllib.request.urlretrieve(url, dest_path, reporthook)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def extract_archive(archive_path: str, extract_dir: str):
    """Extracts a zip or tar.gz archive."""
    if not os.path.exists(archive_path):
        print(f"Archive {archive_path} not found.")
        return

    print(f"Extracting {archive_path} to {extract_dir}...")
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        print(f"Unsupported archive format: {archive_path}")

def download_codenet(output_dir: str):
    """Downloads a sample of Project CodeNet."""
    os.makedirs(output_dir, exist_ok=True)
    # Using a smaller placeholder URL for demonstration. 
    # The real CodeNet is 14M samples and massive.
    url = "https://raw.githubusercontent.com/IBM/Project_CodeNet/main/README.md"
    dest = os.path.join(output_dir, "CodeNet_README.md")
    download_file(url, dest)
    print("Project CodeNet partial download complete.")

def download_codexglue(output_dir: str):
    """Downloads CodeXGLUE defect detection dataset."""
    os.makedirs(output_dir, exist_ok=True)
    # Using raw links to the dataset for demonstration
    base_url = "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/Defect-detection/dataset/"
    for file in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
        url = base_url + file
        dest = os.path.join(output_dir, file)
        download_file(url, dest)
    print("CodeXGLUE download complete.")

def download_juliet(output_dir: str):
    """Downloads Juliet Test Suite (C/C++)."""
    os.makedirs(output_dir, exist_ok=True)
    url = "https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip"
    dest = os.path.join(output_dir, "juliet_c.zip")
    download_file(url, dest)
    extract_archive(dest, os.path.join(output_dir, "juliet"))

def create_sample_dataset(output_dir: str, n_samples: int = 100):
    """Creates synthetic code samples for testing pipelines."""
    os.makedirs(output_dir, exist_ok=True)
    import json
    import random
    
    samples = []
    for i in range(n_samples):
        is_buggy = random.random() < 0.3
        
        if is_buggy:
            code = f"""def calculate_{i}(data):
    # Missing input validation
    result = eval(data)
    for j in range({random.randint(1, 100)}):
        result += j
    return result
"""
        else:
            code = f"""def calculate_{i}(data):
    if not isinstance(data, (int, float)):
        return 0
    result = data
    for j in range({random.randint(1, 100)}):
        result += j
    return result
"""
        samples.append({
            "project": "synthetic",
            "commit_id": f"syn_{i:04d}",
            "target": 1 if is_buggy else 0,
            "func": code
        })
        
    out_file = os.path.join(output_dir, "synthetic_dataset.jsonl")
    with open(out_file, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
            
    print(f"Created {n_samples} synthetic samples at {out_file}")

if __name__ == "__main__":
    base_dir = "data"
    create_sample_dataset(base_dir, 200)
    # download_codexglue(os.path.join(base_dir, "codexglue"))
