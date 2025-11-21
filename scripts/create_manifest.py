import os
import pandas as pd

DATA_DIR = "data/raw/faces_aligned_small_mirrored_co_aligned_cropped_cleaned"
OUTPUT_CSV = "data/metadata.csv"

def create_manifest():
    data = []
    
    for gender_dir in ["F", "M"]:
        dir_path = os.path.join(DATA_DIR, gender_dir)
        if not os.path.exists(dir_path):
            continue
            
        for filename in os.listdir(dir_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
                
            # Parse year from filename (e.g., 1905_Ohio_Cleveland_Central_0-1.png)
            try:
                year = int(filename.split('_')[0])
            except ValueError:
                print(f"Skipping {filename}: Could not parse year")
                continue
                
            # Create relative path from project root
            filepath = os.path.join(DATA_DIR, gender_dir, filename).replace("\\", "/")
            
            data.append({
                "filepath": filepath,
                "year": year
            })
            
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Manifest saved to {OUTPUT_CSV} with {len(df)} entries.")
    print("First 5 rows:")
    print(df.head())

if __name__ == "__main__":
    create_manifest()
