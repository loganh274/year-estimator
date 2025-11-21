import pandas as pd

df = pd.read_csv("data/metadata.csv")
print(f"Min Year: {df['year'].min()}")
print(f"Max Year: {df['year'].max()}")
print(f"Mean Year: {df['year'].mean()}")
print(f"Std Year: {df['year'].std()}")
