"""
grid_search.py

Script to perform a grid search for the best regression model hyperparameters using GradientBoostingRegressor.
Parameters are passed via command-line arguments.

Use:
    python grid_search.py --input_path data/processed_data_norm --output_path models --n_estimators 50 --learning_rate 0.1 --max_depth 5
"""

import argparse
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

def main(input_path: str, output_path: str, n_estimators: int, learning_rate: float, max_depth: int):
    # Load the scaled training data from the normalized folder
    X_train_scaled = pd.read_csv(f"{input_path}/X_train_scaled.csv")
    # Load y_train from the processed_data folder
    y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze("columns")
    
    # Define parameter grid based on CLI inputs (as single-element lists)
    param_grid = {
        "n_estimators": [n_estimators],
        "learning_rate": [learning_rate],
        "max_depth": [max_depth]
    }
    
    # Initialize the GradientBoostingRegressor
    gbr = GradientBoostingRegressor(random_state=42)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=gbr,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X_train_scaled, y_train)
    
    # Retrieve best parameters and save as .pkl
    best_params = grid_search.best_params_
    with open(f"{output_path}/best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)
    
    print("GridSearch completed. Best parameters saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform a grid search for the best regression model parameters using GradientBoostingRegressor."
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to scaled training data (X_train_scaled.csv).")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the best_params.pkl.")
    parser.add_argument("--n_estimators", type=int, required=True, help="Number of estimators to test (e.g., 50).")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate to test (e.g., 0.1).")
    parser.add_argument("--max_depth", type=int, required=True, help="Max depth to test (e.g., 5).")
    
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.n_estimators, args.learning_rate, args.max_depth)
