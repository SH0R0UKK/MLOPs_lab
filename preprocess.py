import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Load the raw data
print("Loading raw data...")
df = pd.read_csv('data/wine_data_raw.csv')

def preprocess_data(df):
    # 1. Check for missing values and print initial info
    print("\nInitial data shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\n NaN values :\n", df.isna().sum())
    
    # 2. Handle missing values if any
    df = df.dropna()  # Remove rows with NaN values
    print("\nShape after removing NaN values:", df.shape)
    
    # 3. Remove duplicates
    df = df.drop_duplicates()
    print("Shape after removing duplicates:", df.shape)
    
    return df


# Perform preprocessing
print("\nPerforming preprocessing steps...")
processed_df = preprocess_data(df)

# Split the data
print("\nSplitting data into train and test sets...")
X = processed_df.drop('quality', axis=1)
y = processed_df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save processed datasets
print("\nSaving processed data...")
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

# Print basic statistics
print("\nDataset shapes:")
print(f"Training set: {train_data.shape}")
print(f"Test set: {test_data.shape}")