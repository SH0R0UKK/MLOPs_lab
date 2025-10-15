import pandas as pd
import os

# URL for the Wine Quality dataset (white wine variant)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
df = pd.read_csv(url, sep=';')

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Save the raw data
df.to_csv('data/wine_data_raw.csv', index=False)

# Quick dataset info (optional)
print("Dataset shape:", df.shape)
print("\nFeatures in the dataset:")
print(df.columns.tolist())