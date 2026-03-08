# MLOps Fraud Detection Pipeline

An end-to-end machine learning operations (MLOps) pipeline built on AWS that automatically detects credit card fraud using a fully automated training, evaluation, deployment, and monitoring workflow. The entire infrastructure is defined as code using AWS CloudFormation.

## Architecture Overview

New transaction data lands in S3 → triggers an automated pipeline → data is processed and cleaned → model retrains on fresh data → model is evaluated against performance thresholds → if it passes, it deploys automatically to a live endpoint → monitoring watches for data drift and performance degradation → if quality drops, the cycle triggers again.

## Tech Stack

- **Language:** Python
- **ML Framework:** scikit-learn, XGBoost via Amazon SageMaker
- **Infrastructure:** AWS CloudFormation (Infrastructure as Code)
- **AWS Services:** S3, SageMaker (Pipelines, Endpoints, Model Registry, Model Monitor), Lambda, EventBridge, Glue, CloudWatch, SNS, CodePipeline, ECR
- **Dataset:** Credit Card Fraud Detection (284,807 transactions, binary classification)

## Setup

Install project dependencies (includes CloudFormation linting tool):

```bash
pip install -r requirements.txt
```

Deploy the data layer stack:

```bash
./infrastructure/deploy.sh
```

Note: activating `.venv` is optional for deployment. The script auto-detects `.venv/bin/cfn-lint` when available.

## Key Features

- Fully automated retraining triggered by new data arrival or scheduled intervals
- SageMaker Pipeline with built-in evaluation gates that prevent underperforming models from reaching production
- Model versioning and approval workflow through SageMaker Model Registry
- Auto-scaling SageMaker Endpoint for real-time inference
- Data drift detection and model performance monitoring with automated alerting
- CI/CD pipeline for infrastructure and ML code changes
- Modular CloudFormation nested stacks (data layer, training, deployment, monitoring)

## Project Status

🔨 Currently in development building incrementally phase by phase.

- [x] Phase 1: Data layer (S3, Glue, data exploration and preprocessing)
- [ ] Phase 2: Training pipeline (SageMaker Pipeline with evaluation gates)
- [ ] Phase 3: Automated deployment (Lambda-triggered endpoint updates)
- [ ] Phase 4: Trigger mechanisms (S3 events, EventBridge schedules)
- [ ] Phase 5: Monitoring and drift detection (Model Monitor, CloudWatch)
- [ ] Phase 6: Full IaC (all infrastructure in CloudFormation)
- [ ] Phase 7: CI/CD meta-pipeline (CodePipeline for infrastructure deployment)
