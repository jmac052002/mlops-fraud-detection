#!/bin/bash
set -euo pipefail

# ============================================================
# MLOps Fraud Detection Pipeline - Full Deployment Script
# Deploys all three CloudFormation stacks in correct order
# Usage: bash infrastructure/deploy.sh
# ============================================================

PROJECT_NAME="mlops-fraud-detection"
ENVIRONMENT="dev"
PIPELINE_NAME="mlops-fraud-detection-pipeline"
ALERT_EMAIL="jmac052002@gmail.com"

# Template paths
DATA_TEMPLATE="infrastructure/data-layer.yaml"
TRIGGER_TEMPLATE="infrastructure/templates/trigger-layer.yaml"
MONITORING_TEMPLATE="infrastructure/templates/monitoring-layer.yaml"

# Stack names
DATA_STACK="${PROJECT_NAME}-data-layer"
TRIGGER_STACK="${PROJECT_NAME}-trigger"
MONITORING_STACK="${PROJECT_NAME}-monitoring"

# ============================================================
# Validate all templates with cfn-lint before deploying
# ============================================================
if [[ -x ".venv/bin/cfn-lint" ]]; then
  CFN_LINT_BIN=".venv/bin/cfn-lint"
elif command -v cfn-lint >/dev/null 2>&1; then
  CFN_LINT_BIN="$(command -v cfn-lint)"
else
  echo "[ERROR] cfn-lint not found. Run: pip install -r requirements.txt"
  exit 1
fi

echo "============================================================"
echo "Validating CloudFormation templates..."
echo "============================================================"
"$CFN_LINT_BIN" "$DATA_TEMPLATE" "$TRIGGER_TEMPLATE" "$MONITORING_TEMPLATE"
echo "All templates valid."

# ============================================================
# Step 1: Deploy data layer (S3 bucket + SageMaker role)
# ============================================================
echo ""
echo "============================================================"
echo "Step 1/3: Deploying data layer stack..."
echo "============================================================"
aws cloudformation deploy \
  --template-file "$DATA_TEMPLATE" \
  --stack-name "$DATA_STACK" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProjectName="$PROJECT_NAME" \
    Environment="$ENVIRONMENT"
echo "Data layer deployed."

# Resolve bucket name from CloudFormation exports
DATA_BUCKET_NAME="$(aws cloudformation list-exports \
  --query "Exports[?Name=='${PROJECT_NAME}-data-bucket'].Value | [0]" \
  --output text)"

if [[ -z "$DATA_BUCKET_NAME" || "$DATA_BUCKET_NAME" == "None" ]]; then
  echo "[ERROR] Could not resolve data bucket export: ${PROJECT_NAME}-data-bucket"
  exit 1
fi
echo "Resolved bucket: ${DATA_BUCKET_NAME}"

# ============================================================
# Step 2: Deploy trigger layer (Lambda + EventBridge S3 rule)
# ============================================================
echo ""
echo "============================================================"
echo "Step 2/3: Deploying trigger layer stack..."
echo "============================================================"
aws cloudformation deploy \
  --template-file "$TRIGGER_TEMPLATE" \
  --stack-name "$TRIGGER_STACK" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    PipelineName="$PIPELINE_NAME" \
    DataBucketName="$DATA_BUCKET_NAME"
echo "Trigger layer deployed."

# ============================================================
# Step 3: Deploy monitoring layer (SNS + monitoring Lambda)
# ============================================================
echo ""
echo "============================================================"
echo "Step 3/3: Deploying monitoring layer stack..."
echo "============================================================"
aws cloudformation deploy \
  --template-file "$MONITORING_TEMPLATE" \
  --stack-name "$MONITORING_STACK" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    PipelineName="$PIPELINE_NAME" \
    AlertEmail="$ALERT_EMAIL" \
    DataBucketName="$DATA_BUCKET_NAME"
echo "Monitoring layer deployed."

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "Deployment complete!"
echo "Stacks deployed:"
echo "  - ${DATA_STACK}"
echo "  - ${TRIGGER_STACK}"
echo "  - ${MONITORING_STACK}"
echo ""
echo "Next steps:"
echo "  Run pipeline: python src/pipeline/sagemaker_pipeline.py"
echo "  Test trigger: aws s3 cp data/raw/creditcard.csv s3://${DATA_BUCKET_NAME}/raw/test.csv"
echo "============================================================"