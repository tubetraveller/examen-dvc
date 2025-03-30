"""
model_training.py

Script to train a regression model using the best parameters found by GridSearch with GradientBoostingRegressor.

Use:
    python model_training.py --input_path data/processed_data_norm --params_path models/best_params.pkl --output_path models
"""

import argparse
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

def main(input_path: str, params_path: str, output_path: str):
    # Load scaled train data from the normalized folder
    X_train_scaled = pd.read_csv(f"{input_path}/X_train_scaled.csv")
    # Load y_train from the original processed_data folder (unchanged, not normalized)
    y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze("columns")
    
    # Load best parameters
    with open(params_path, "rb") as f:
        best_params = pickle.load(f)
    
    # Initialize model with best parameters
    model = GradientBoostingRegressor(**best_params, random_state=42)
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Save the trained model
    with open(f"{output_path}/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Model training completed. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a regression model with best parameters using GradientBoostingRegressor."
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to scaled X_train (from data/processed_data_norm).")
    parser.add_argument("--params_path", type=str, required=True, help="Path to best_params.pkl.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the trained model.")
    
    args = parser.parse_args()
    main(args.input_path, args.params_path, args.output_path)