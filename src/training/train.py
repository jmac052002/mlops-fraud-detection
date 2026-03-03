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
    parser.add_argument(
        "--train-dir", 
        type=str, 
        required=True, 
        help="Directory containing train.csv"
    )
    parser.add_argument(
        "--val-dir", 
        type=str, 
        required=True, 
        help="Directory containing validation.csv"
    )
    parser.add_argument(
        "--test-dir", 
        type=str, 
        required=True, 
        help="Directory containing test.csv"
    )

    # Output directory for the model artifact
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default="models/", 
        help="Directory where the trained model will be saved (default: models/)"
    )

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

    



if __name__ == "__main__":
    args = parse_args()
    
    logger.info("Arguments received:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}") 
 
        

    