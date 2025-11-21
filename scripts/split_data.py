import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

METADATA_CSV = "data/metadata.csv"
TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"

def split_dataset():
    df = pd.read_csv(METADATA_CSV)
    
    # Filter out years with fewer than 2 samples
    year_counts = df['year'].value_counts()
    valid_years = year_counts[year_counts >= 2].index
    df = df[df['year'].isin(valid_years)]
    
    print(f"Filtered {len(year_counts) - len(valid_years)} years with < 2 samples. Remaining images: {len(df)}")

    # Stratified split
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['year'], 
        random_state=42
    )
    
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    
    print(f"Train set: {len(train_df)} images saved to {TRAIN_CSV}")
    print(f"Val set: {len(val_df)} images saved to {VAL_CSV}")
    
    # Validation: Check distributions
    print("\nYear Distribution Summary:")
    print(f"Train Year Mean: {train_df['year'].mean():.2f}, Std: {train_df['year'].std():.2f}")
    print(f"Val Year Mean: {val_df['year'].mean():.2f}, Std: {val_df['year'].std():.2f}")
    
    # Simple text-based histogram check
    print("\nDecade Distribution Check (First 5 decades):")
    train_decades = (train_df['year'] // 10) * 10
    val_decades = (val_df['year'] // 10) * 10
    
    train_counts = train_decades.value_counts(normalize=True).sort_index().head(5)
    val_counts = val_decades.value_counts(normalize=True).sort_index().head(5)
    
    print(f"{'Decade':<10} {'Train %':<10} {'Val %':<10}")
    for decade in train_counts.index:
        print(f"{decade:<10} {train_counts.get(decade, 0):.4f}     {val_counts.get(decade, 0):.4f}")

if __name__ == "__main__":
    split_dataset()
