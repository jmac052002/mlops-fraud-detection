import argparse 
import os 
import logging 
import json 
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score 
import joblib 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") 
logger = logging.getLogger(__name__) 

def parse_args():
    """
    Parses arguments for training the Logistic Regression model.
    """
    parser = argparse.ArgumentParser(description="Train a Logistic Regression model on SageMaker")

    # Data directories (where SageMaker will mount S3 data)
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "data/processed/train"))
    parser.add_argument("--val-dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "data/processed/validation"))
    parser.add_argument("--test-dir", type=str, default=os.environ.get("SM_CHANNEL_TEST", "data/processed/test"))
    
    # Output directory for the model artifact 
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "models"))
    
    # Model Hyperparameters 
    parser.add_argument(
        "--max-iter", 
        type=int, 
        default=1000, 
        help="Maximum iterations for Logistic Regression solver (default: 1000)"
    )
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility (default: 42)"
    )

    return parser.parse_args() 

def load_data(data_dir, dataset_name):
    """
    Loads the dataset from the specified directory.
    """
    file_path = os.path.join(data_dir, f"{dataset_name}.csv")
    logger.info(f"Loading {dataset_name} data from {file_path}")
    # 2. Read the CSV file into a DataFrame 
    df = pd.read_csv(file_path) 

    # 3. Seperate Features (X) and Target (y) 
    X = df.drop(columns=['Class']) 
    y = df['Class'] 

    # 4. Log the shape and class distribution (Fraud vs. Non-Fraud)
    logger.info(f"{dataset_name} set loaded. Shape: {X.shape}") 

    # Using value_counts to see the balance of 0s and 1s
    class_counts = y.value_counts().to_dict()
    logger.info(f"{dataset_name} class distribution: {class_counts}")
    
    # 5. Return X and y as separate objects
    return X, y 

    if not os.path.exists(file_path): 
        raise FileNotFoundError(f"Critical Error: {file_path} not found!") 

def train_model(X_train, y_train, max_iter, random_state):
    """
    Trains a Logistic Regression model with the given training data and hyperparameters.
    """
    logger.info("Training Logistic Regression model...") 
    # 1. Initialize the model with the hyperparameters from parse_args 
    model = LogisticRegression(max_iter=max_iter, random_state=random_state) 
    # 2. Fit the model on the training data
    # This is the step where the model learns the weights for each feature 
    model.fit(X_train, y_train) 
    logger.info("Model training completed.") 
    # 3. Return the trained model object 
    return model 

def evaluate_model(model, X, y, dataset_name):
    """
    Evaluates the model and returns a dictionary of key performance metrics.
    """
    logger.info(f"--- Evaluation for {dataset_name} set ---")

    # 1. Generate predictions (0 or 1)
    y_pred = model.predict(X)

    # 2. Generate probability scores for the positive class (Fraud)
    # predict_proba returns [prob_0, prob_1], so we take the second column [:, 1]
    y_prob = model.predict_proba(X)[:, 1]

    # 3. Calculate individual metrics
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)

    # 4. Build the metrics dictionary
    metrics = {
        "dataset": dataset_name,
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "roc_auc": round(roc_auc, 4)
    }

    # 5. Log the Confusion Matrix and Metrics
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")

    # 6. Log the full Classification Report (Precision, Recall, F1 for both classes)
    logger.info(f"Classification Report:\n{classification_report(y, y_pred)}")

    return metrics 

def save_model(model, metrics, model_dir):
    """
    Saves the trained model and metrics to the specified directory.
    """
    # 1. Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # 2. Save the model using joblib
    model_path = os.path.join(model_dir, "logistic_regression_model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # 3. Save the metrics to a JSON file
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}") 

def main(): 
    args = parse_args() 

    logger.info("Arguments received:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    logger.info("Starting Training pipeline...") 

    # 1. Load data 
    X_train, y_train = load_data(args.train_dir, "train") 
    X_val, y_val = load_data(args.val_dir, "validation") 
    X_test, y_test = load_data(args.test_dir, "test") 

    # 2. Train the model 
    model = train_model(X_train, y_train, args.max_iter, args.random_state) 

    # 3. Evaluate on all three sets 
    train_metrics = evaluate_model(model, X_train, y_train, "Train") 
    val_metrics = evaluate_model(model, X_val, y_val, "Validation") 
    test_metrics = evaluate_model(model, X_test, y_test, "Test") 

    # 4. Save the model and metrics 
    save_model(model, test_metrics, args.model_dir) 

    logger.info("Training pipeline completed successfully.")  


if __name__ == "__main__":
    main()
    
   
        

    