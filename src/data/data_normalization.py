"""
data_normalization.py

Script to normalize the train and test sets using a chosen scaler (e.g. StandardScaler).

Use:
    python data_normalization.py --input_path data/processed_data --output_path data/processed_data
"""

import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main(input_path: str, output_path: str):
    # Read the train and test features
    X_train = pd.read_csv(f"{input_path}/X_train.csv")
    X_test = pd.read_csv(f"{input_path}/X_test.csv")
    
    # Fit a StandardScaler on X_train
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Transform train and test sets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame with original column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Save scaled data
    X_train_scaled.to_csv(f"{output_path}/X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(f"{output_path}/X_test_scaled.csv", index=False)
    
    print("Data normalization completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize the train and test sets.")
    parser.add_argument("--input_path", type=str, required=True, help="Path where X_train.csv and X_test.csv are stored.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the normalized data.")
    
    args = parser.parse_args()
    main(args.input_path, args.output_path)
