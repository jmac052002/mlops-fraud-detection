import argparse
import os
import logging
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE 
import joblib 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelnames)s - %(message)s") 
logger = logging.getLogger(__name__) 

def parse_args(): 
    parser = argparse.ArgumentParser(description="Process data for machine learning") 
    parser.add_argument("--input-data", type=str, required=True, help="Path to raw CSV") 
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the preprocessed data and scaler") 
    parser.add_argument("--test-size", type=float, default=0.15, help="Proportion of the dataset to include in the test split") 
    parser.add_argument("--val-size", type=float, default=0.15, help="Proportion of data for the validation set") 
    parser.add_argument("--random-state", type=int, default=42, help="Seed for reproducibility") 
    return parser.parse_args() 

def load_data(file_path): 
    """
    Reads a CSV file and logs the process using the logging module.
    """
    try: 
        # 1. Attempt to read the CSV 
        df = pd.read_csv(file_path) 

        # 2. Log success and the shape of the data 
        logging.info(f"Successfully loaded data from {file_path}") 
        logging.info(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns") 
        return df
    except Exception as e: 
        # 3. Log an error if the file is missing or corrupted
        logging.error(f"Error loading data from {file_path}: {e}")
        raise # This stops the script so you don't process an empty variable 

def split_data(df, test_size, val_size, random_state): 
    """
    Splits the raw DataFrame into Train, Validation, and Test sets. 
    """ 
    # 1. Seperate Features (X) and Target (y) 
    X = df.drop(columns=['Class']) 
    y = df['Class'] 

    # 2. First Split: Seperate the Test set from everything else
    # Stratify=y ensures the ratio of fraud cases is the same in both sets 
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y) 

    # 3. Second Split: Split the remaining data into Train and Validation
    # We adjust the ratio because the "temp" data is already smaller than the original
    val_ratio = val_size / (1 - test_size) 
    
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state, stratify=y_temp)
    
    # 4. Log the shapes and fraud percentages 
    datasets = {
        "Train": (X_train, y_train), 
        "Val": (X_val, y_val), 
        "Test": (X_test, y_test)    
    } 

    for name, (feat, target) in datasets.items(): 
        fraud_pct = (target.sum() / len(target)) * 100 
        logging.info(f"{name} set: {feat.shape[0]} rows | Fraud: {fraud_pct:.4f}%") 

    return X_train, X_val, X_test, y_train, y_val, y_test 

def scale_features(X_train, X_val, X_test, output_dir): 
    """
     Standardizes features using StandardScaler. 
    Fits only on training data to prevent leakage.
    """ 
    scaler = StandardScaler() 

    # 1. Fit and transform the training set 
    X_train_scaled = scaler.fit_transform(X_train) 

    # 2. Transform (DO NOT FIT) validation and test sets 
    X_val_scaled = scaler.transform(X_val) 
    X_test_scaled = scaler.transform(X_test) 

    # 3. Save the scaler for future use (inference) 
    scaler_path = os.path.join(output_dir, "scaler.joblib") 
    joblib.dump(scaler, scaler_path) 
    logging.info(f"Scaler saved to {scaler_path}") 

    return X_train_scaled, X_val_scaled, X_test_scaled 




        
        
   








    












































if __name__ == "__main__": 
    main() 
    args = parse_args() 
