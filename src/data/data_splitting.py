"""
data_splitting.py

Script to split the flotation dataset into train and test sets.
We ignore the 'date' column for prediction, and assume 'silica_concentrate'
is the last column in the dataset.

Use:
    python data_splitting.py --input_path data/raw_data/raw.csv --output_path data/processed_data --test_size 0.2 --random_state 42
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main(input_path: str, output_path: str, test_size: float, random_state: int):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Read the dataset
    df = pd.read_csv(input_path)
    
    # Drop the 'date' column if it exists
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    # Separate features (X) and target (y) (assuming the last column is the target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Save the resulting splits
    X_train.to_csv(os.path.join(output_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_path, "y_test.csv"), index=False)
    
    print("Data splitting completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the data into train and test sets.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the raw CSV file.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to store the split files.")
    parser.add_argument("--test_size", type=float, required=True, help="Test set proportion (e.g., 0.2).")
    parser.add_argument("--random_state", type=int, required=True, help="Random seed for reproducibility (e.g., 42).")
    
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.test_size, args.random_state)

