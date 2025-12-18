import os
import csv
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_ROOT = os.path.abspath("./data")
SOURCE_FOLDER = os.path.join(DATA_ROOT, "DR_grading") # Where the messy user data is
CSV_PATH = os.path.join(DATA_ROOT, "DR_grading.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
VAL_SPLIT = 0.2
SEED = 42

def clean_filename(fname):
    return fname.strip()

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    # 1. Create Train/Val structure
    for split in [TRAIN_DIR, VAL_DIR]:
        for i in range(5):
            os.makedirs(os.path.join(split, str(i)), exist_ok=True)

    # 2. Index all available images
    print("Scanning for images...")
    image_map = {}
    
    # Recursively find all jpg/png files in SOURCE_FOLDER
    source_path = Path(SOURCE_FOLDER)
    files = list(source_path.rglob("*.jpg")) + list(source_path.rglob("*.png")) + list(source_path.rglob("*.jpeg"))
    
    print(f"Found {len(files)} potential images in {SOURCE_FOLDER}")
    
    for f in files:
        image_map[f.name] = str(f)

    # 3. Process CSV
    print("Processing CSV matching...")
    moved_count = 0
    missing_count = 0
    
    random.seed(SEED)
    
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # Shuffle to ensure random validation split
        random.shuffle(rows)
        
        # Split index
        val_cutoff = int(len(rows) * VAL_SPLIT)
        
        for idx, row in enumerate(tqdm(rows)):
            filename = clean_filename(row['id_code'])
            label = row['diagnosis']
            
            # Determine split (this is a robust way to split even if we skip files)
            # Actually, better to decide split per-file to maintain ratio effectively 
            # but simple random assignment is fine for large datasets.
            is_val = (random.random() < VAL_SPLIT)
            dest_root = VAL_DIR if is_val else TRAIN_DIR
            
            if filename in image_map:
                src_path = image_map[filename]
                dest_path = os.path.join(dest_root, label, filename)
                
                # Check if already processed
                if os.path.exists(dest_path):
                    continue
                    
                shutil.copy2(src_path, dest_path)
                moved_count += 1
            else:
                # This is normal if the CSV contains other datasets (like IDRiD) not present in the folder
                missing_count += 1

    print(f"\ndataset reorganization complete.")
    print(f"Successfully organized {moved_count} images.")
    print(f"Skipped {missing_count} entries (files not found in folder).")
    print(f"Train/Val split created in: {TRAIN_DIR} and {VAL_DIR}")

if __name__ == "__main__":
    main()
