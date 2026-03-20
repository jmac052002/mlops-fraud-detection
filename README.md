# MLOps Fraud Detection Pipeline

An end-to-end MLOps pipeline on AWS that automates the full lifecycle of a fraud detection model from raw data ingestion through training, evaluation, quality gating, model registration, deployment, and monitoring. All infrastructure is defined as code using AWS CloudFormation.

---

## Architecture
```
New CSV uploaded to S3
        ↓
EventBridge detects upload → triggers Lambda
        ↓
Lambda starts SageMaker Pipeline
        ↓
PreprocessData → TrainModel → EvaluateModel → CheckModelQuality → RegisterModel
        ↓
Pipeline completes → Monitoring Lambda fires
        ↓
Model metrics published to CloudWatch + Email alert sent via SNS
        ↓
Registered model deployed to SageMaker Endpoint
        ↓
Model Monitor baseline watches for data drift
```

---

## Tech Stack

- **Language:** Python 3.12
- **ML:** scikit-learn (Logistic Regression), imbalanced-learn (SMOTE)
- **Infrastructure:** AWS CloudFormation (3 stacks)
- **AWS Services:** S3, SageMaker Pipelines, SageMaker Model Registry, SageMaker Endpoints, SageMaker Model Monitor, Lambda, EventBridge, SNS, CloudWatch, IAM
- **Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 284,807 transactions, 0.17% fraud rate

---

## Project Structure
```
mlops-fraud-detection/
├── infrastructure/
│   ├── data-layer.yaml          # S3 bucket, SageMaker IAM role
│   ├── deploy.sh                # Single-command full deployment
│   └── templates/
│       ├── trigger-layer.yaml   # Lambda + EventBridge S3 trigger
│       └── monitoring-layer.yaml # SNS alerts + monitoring Lambda
├── src/
│   ├── processing/
│   │   └── preprocess.py        # Split → Scale → SMOTE pipeline
│   ├── training/
│   │   └── train.py             # Logistic Regression training
│   ├── evaluation/
│   │   └── evaluate.py          # F1, Precision, Recall, ROC-AUC
│   ├── pipeline/
│   │   └── sagemaker_pipeline.py # Full SageMaker Pipeline definition
│   ├── trigger/
│   │   └── lambda_handler.py    # S3 event → pipeline trigger
│   └── deployment/
│       ├── deploy_endpoint.py   # Deploy model to endpoint
│       ├── inference.py         # Endpoint inference handler
│       └── setup_model_monitor.py # Model Monitor baseline + schedule
├── notebooks/
│   └── 01-data-exploration.ipynb
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Deploy all infrastructure
```bash
bash infrastructure/deploy.sh
```
This validates and deploys all three CloudFormation stacks in the correct order.

### 3. Run the pipeline
```bash
python src/pipeline/sagemaker_pipeline.py
```

### 4. Test the S3 trigger
```bash
aws s3 cp data/raw/creditcard.csv \
  s3://mlops-fraud-detection-dev-584996267165/raw/creditcard_test.csv
```

### 5. Deploy the endpoint
```bash
python src/deployment/deploy_endpoint.py
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| F1 Score | 0.1111 |
| Precision | 0.0593 |
| Recall | 0.8784 |
| ROC-AUC | 0.9612 |

> High recall is the design goal catching 88% of fraud cases is more valuable than avoiding false positives in this domain.

---

## Monitoring & Alerting

- **Pipeline alerts:** SNS email on every pipeline success or failure
- **Model metrics:** F1, Precision, Recall, ROC-AUC published to CloudWatch `MLOps/FraudDetection` namespace after every run
- **Data drift:** Model Monitor baseline established from training data constraints and statistics saved to S3

---

## Project Status

- [x] Phase 1: Data layer-S3, IAM, CloudFormation
- [x] Phase 2: Production scripts-preprocessing, training, evaluation
- [x] Phase 3: SageMaker Pipeline-5-step pipeline with quality gate
- [x] Phase 4: Automated triggers-S3 → EventBridge → Lambda → Pipeline
- [x] Phase 5A: Pipeline alerting-SNS email on success/failure
- [x] Phase 5B: CloudWatch metrics-custom namespace with model metrics
- [x] Phase 5C: Model deployment + Model Monitor baseline
- [x] Phase 6: Full IaC-single deploy.sh command, EventBridge fix
- [ ] Phase 7: CI/CD-CodePipeline triggered by GitHub pushes

---

## Repository

[github.com/jmac052002/mlops-fraud-detection](https://github.com/jmac052002/mlops-fraud-detection)
# Phase 7 complete

test
test2

