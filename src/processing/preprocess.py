import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])
import argparse
import os
import logging
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import RobustScaler 
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE 
import joblib 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") 
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
    # If a directory is passed, find the first CSV inside it
    if os.path.isdir(file_path):
        csv_files = [f for f in os.listdir(file_path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in directory: {file_path}")
        file_path = os.path.join(file_path, csv_files[0])
    
    logger.info(f"Loading data from: {file_path}")
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
    
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp) 
    
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


def scale_features(X_train, X_val, X_test):
    """
    Scales 'Amount' and 'Time' using RobustScaler.
    Fits on Train, Transforms on Val and Test to avoid leakage.
    """
    # 1. Initialize the scaler
    scaler = RobustScaler()

    # Define the columns to scale
    cols_to_scale = ['Amount', 'Time']
    new_cols = ['scaled_amount', 'scaled_time']

    # 2. Fit and Transform the Training data
    # We use .values if needed, but assigning back to new columns works directly on DataFrames
    X_train[new_cols] = scaler.fit_transform(X_train[cols_to_scale])

    # 3. Transform (only) Validation and Test data
    X_val[new_cols] = scaler.transform(X_val[cols_to_scale])
    X_test[new_cols] = scaler.transform(X_test[cols_to_scale])

    # 4. Drop the original unscaled columns
    X_train = X_train.drop(columns=cols_to_scale)
    X_val = X_val.drop(columns=cols_to_scale)
    X_test = X_test.drop(columns=cols_to_scale)

    logging.info(f"Features scaled and original '{cols_to_scale}' columns removed.")
    
    # 5. Return the modified DataFrames and the scaler object
    return X_train, X_val, X_test, scaler


def apply_smote(X_train, y_train, random_state):
    """
    Applies SMOTE to the training data to handle class imbalance.
    Only applied to Training data to prevent data leakage.
    """
    # 1. Log class distribution BEFORE SMOTE
    before_counts = y_train.value_counts().to_dict()
    logging.info(f"Class distribution BEFORE SMOTE: {before_counts}")

    # 2. Initialize and Run SMOTE
    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # 3. Log class distribution AFTER SMOTE
    after_counts = y_train_res.value_counts().to_dict()
    logging.info(f"Class distribution AFTER SMOTE: {after_counts}") 

    # 4. Return the resampled (balanced) data
    return X_train_res, y_train_res


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, scaler, output_dir): 
    """
    Saves the processed DataFrames to CSVs in organized directories and exports 
    the scaler for inference.  
    """ 
    # 1. Create subdirectories 
    splits = ['train', 'validation', 'test'] 
    for split in splits: 
        os.makedirs(os.path.join(output_dir, split), exist_ok=True) 

    # 2. Define a list of tuples to iterate through for saving 
    data_to_save = [ 
        ('train', X_train, y_train), 
        ('validation', X_val, y_val), 
        ('test', X_test, y_test) 
    ] 

    # 3. Combine X and y, then save 
    for name, X_data, y_data in data_to_save: 
        # Concatenate features and target along columns (axis=1) 
        combined_df = pd.concat([X_data, y_data], axis=1) 

        # Define the file path 
        file_path = os.path.join(output_dir, name, f"{name}.csv") 

        # Save to CSV 
        combined_df.to_csv(file_path, index=False) 

        # Log the action 
        logging.info(f"Saved {name} set to {file_path} | Rows: {len(combined_df)}") 

    # 4. Save the scaler using joblib for later use in inference 
    scaler_path = os.path.join(output_dir, "scaler.joblib") 
    joblib.dump(scaler, scaler_path) 
    logging.info(f"Scaler saved to {scaler_path}") 

def main():
    args = parse_args()
    
    logger.info("Starting preprocessing pipeline...")
    
    # 1. Load
    df = load_data(args.input_data)
    
    # 2. Split FIRST (before scaling)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, args.test_size, args.val_size, args.random_state
    )
    
    # 3. Scale (fit on train only)
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)
    
    # 4. SMOTE (train only)
    X_train, y_train = apply_smote(X_train, y_train, args.random_state)
    
    # 5. Save everything
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test, scaler, args.output_dir)
    
    logger.info("Preprocessing complete!") 

if __name__ == "__main__":
    main()   






        
        
   








    












































