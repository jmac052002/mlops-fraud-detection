import subprocess
import sys 
subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
import argparse 
import os 
import logging 
import json 
import pandas as pd 
import numpy as np 
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix 
import joblib 
import tarfile 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__) 

# This script will be executed in a SageMaker Processing Job, which means it will have access to the trained model and the test dataset through the specified directories. 
# The script will load the model, make predictions on the test dataset, calculate evaluation metrics, and save the results.
def parse_args(): 
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the test dataset.")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), help="Directory where the trained model is stored.")
    parser.add_argument("--test-data-dir", type=str, default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"), help="Directory where the test data is stored.")
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output"), help="Directory where the evaluation results will be saved.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classifying probabilities into binary classes.")
    return parser.parse_args() 

# This function extracts the trained model from the tar.gz file that SageMaker creates when saving the model. 
# It assumes that the model file is named 'model.joblib' after extraction, which is a common convention for scikit-learn models saved with joblib.
def extract_model(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".tar.gz")]
    if not model_files:
        raise FileNotFoundError("No model file found in the specified directory.")
    
    model_path = os.path.join(model_dir, model_files[0])
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall(path=model_dir)
    
    # Find the extracted joblib file
    joblib_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
    if not joblib_files:
        raise FileNotFoundError("No .joblib model file found after extraction.")
    
    model = joblib.load(os.path.join(model_dir, joblib_files[0]))
    logger.info(f"Model loaded: {joblib_files[0]}")
    return model


# Find the CSV file in the test directory, Read it with pandas, 
# Separate X (everything except Class) and y (Class), Log the shape, Return X, y
def load_test_data(test_data_dir): 
    test_files = [f for f in os.listdir(test_data_dir) if f.endswith(".csv")]
    if not test_files:
        raise FileNotFoundError("No test data file found in the specified directory.")
    
    test_data_path = os.path.join(test_data_dir, test_files[0])
    df = pd.read_csv(test_data_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    logger.info(f"Test data shape: X={X.shape}, y={y.shape}")
    return X, y 

# Generate predictions: y_pred = model.predict(X_test)
# Generate probabilities: y_prob = model.predict_proba(X_test)[:, 1]
def evaluate_model(model, X_test, y_test): 
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Assuming binary classification
    return y_pred, y_prob

# Calculate metrics: f1_score, precision_score, recall_score, roc_auc_score 
# Build a dictionary with the metrics and return it.
# Log the metrics, confusion matrix, and classification report for better visibility in the SageMaker logs.
def calculate_metrics(y_test, y_pred, y_prob): 
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    logger.info(f"Evaluation Metrics - F1 Score: {f1}, Precision: {precision}, Recall: {recall}, ROC AUC: {roc_auc}")  
    metrics = {
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "roc_auc": round(roc_auc, 4)
    }
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return metrics 

# Create the output directory: os.makedirs(output_dir, exist_ok=True)
# Write the metrics dict to metrics.json
# Log the path where the metrics are saved for better visibility in the SageMaker logs.
def save_metrics(metrics, output_dir): 
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Evaluation metrics saved to {metrics_path}")

# This function creates the processing step of the pipeline 
# Which will preprocess the raw data and split it into train/validation/test sets.
def main():
    args = parse_args()

    logger.info("Starting model evaluation...")

    # 1. Extract and load the trained model
    model = extract_model(args.model_dir)

    # 2. Load test data
    X_test, y_test = load_test_data(args.test_data_dir)

    # 3. Evaluate
    y_pred, y_prob = evaluate_model(model, X_test, y_test)

    # 4. Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)

    # 5. Log pass/fail against threshold
    if metrics["f1_score"] >= args.threshold:
        logger.info(f"PASSED: F1 score {metrics['f1_score']} >= threshold {args.threshold}")
    else:
        logger.info(f"FAILED: F1 score {metrics['f1_score']} < threshold {args.threshold}")

    # 6. Save metrics
    save_metrics(metrics, args.output_dir)

    logger.info("Evaluation complete!")
    
if __name__ == "__main__":
    main() 