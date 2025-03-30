"""
grid_search.py

Script to perform a grid search for the best regression model hyperparameters using GradientBoostingRegressor.

Use:
    python grid_search.py --input_path data/processed_data_norm --output_path models
"""

import argparse
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

def main(input_path: str, output_path: str):
    # Load the scaled training data from the normalized folder
    X_train_scaled = pd.read_csv(f"{input_path}/X_train_scaled.csv")
    
    # Load y_train from the processed_data folder (fixed path)
    y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze("columns")
    
    # Define parameter grid for GradientBoostingRegressor
    param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    }
    
    # Initialize the model
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
    
    # Retrieve best parameters
    best_params = grid_search.best_params_
    
    # Save best parameters as a .pkl file
    with open(f"{output_path}/best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)
    
    print("GridSearch completed. Best parameters saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform a grid search for the best regression model parameters using GradientBoostingRegressor."
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to scaled train data (X_train_scaled.csv).")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the best_params.pkl.")
    
    args = parser.parse_args()
    main(args.input_path, args.output_path)
