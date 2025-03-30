"""
model_evaluation.py

Script to evaluate the trained model on the test set, save predictions, and record metrics.

Use:
    python model_evaluation.py --model_path models/trained_model.pkl --input_path data/processed_data 
    --metrics_path metrics/scores.json --output_path data/predictions
"""

import argparse
import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score

def main(model_path: str, input_path: str, metrics_path: str, output_path: str):
    # Load the test data
    X_test_scaled = pd.read_csv(f"{input_path}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{input_path}/y_test.csv").squeeze("columns")
    
    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Save predictions as CSV
    predictions_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
    predictions_df.to_csv(f"{output_path}/predictions.csv", index=False)
    
    # Calculate metrics
    mse_value = mean_squared_error(y_test, y_pred)
    r2_value = r2_score(y_test, y_pred)
    
    # Save metrics in JSON
    metrics = {
        "mse": mse_value,
        "r2": r2_value
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("Model evaluation completed. Metrics saved and predictions stored.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained model and save metrics/predictions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to X_test_scaled.csv and y_test.csv.")
    parser.add_argument("--metrics_path", type=str, required=True, help="Path to save the evaluation metrics (JSON).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save predictions.csv.")
    
    args = parser.parse_args()
    main(args.model_path, args.input_path, args.metrics_path, args.output_path)
