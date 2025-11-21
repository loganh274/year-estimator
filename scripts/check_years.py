import pandas as pd

METADATA_CSV = "data/metadata.csv"

def check_counts():
    df = pd.read_csv(METADATA_CSV)
    counts = df['year'].value_counts().sort_values()
    print("Years with fewer than 2 samples:")
    print(counts[counts < 2])
    print("\nYears with fewer than 5 samples:")
    print(counts[counts < 5])

if __name__ == "__main__":
    check_counts()
