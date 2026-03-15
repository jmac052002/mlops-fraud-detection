#!/bin/bash
set -euo pipefail

PROJECT_NAME="mlops-fraud-detection"
ENVIRONMENT="dev"

DATA_STACK_NAME="${PROJECT_NAME}-data-layer"
DATA_TEMPLATE="infrastructure/data-layer.yaml"

MONITORING_STACK_NAME="${PROJECT_NAME}-monitoring"
MONITORING_TEMPLATE="infrastructure/templates/monitoring-layer.yaml"
PIPELINE_NAME="mlops-fraud-detection-pipeline"
ALERT_EMAIL="jmac052002@gmail.com"

if [[ -x ".venv/bin/cfn-lint" ]]; then
  CFN_LINT_BIN=".venv/bin/cfn-lint"
elif command -v cfn-lint >/dev/null 2>&1; then
  CFN_LINT_BIN="$(command -v cfn-lint)"
else
  echo "[cfn-lint] cfn-lint not found. Install dependencies with: pip install -r requirements.txt"
  echo "More help: https://github.com/aws-cloudformation/cfn-python-lint/#install"
  exit 1
fi

echo "Running cfn-lint on ${DATA_TEMPLATE} and ${MONITORING_TEMPLATE}..."
"$CFN_LINT_BIN" "$DATA_TEMPLATE" "$MONITORING_TEMPLATE"

echo "Deploying data layer stack (${DATA_STACK_NAME})..."
aws cloudformation deploy \
  --template-file "$DATA_TEMPLATE" \
  --stack-name "$DATA_STACK_NAME" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProjectName="$PROJECT_NAME" \
    Environment="$ENVIRONMENT"

DATA_BUCKET_NAME="$(aws cloudformation list-exports \
  --query "Exports[?Name=='${PROJECT_NAME}-data-bucket'].Value | [0]" \
  --output text)"

if [[ -z "$DATA_BUCKET_NAME" || "$DATA_BUCKET_NAME" == "None" ]]; then
  echo "Could not resolve data bucket export: ${PROJECT_NAME}-data-bucket"
  exit 1
fi

echo "Resolved DataBucketName=${DATA_BUCKET_NAME}"
echo "Deploying monitoring stack (${MONITORING_STACK_NAME})..."
aws cloudformation deploy \
  --template-file "$MONITORING_TEMPLATE" \
  --stack-name "$MONITORING_STACK_NAME" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    PipelineName="$PIPELINE_NAME" \
    AlertEmail="$ALERT_EMAIL" \
    DataBucketName="$DATA_BUCKET_NAME"

echo "Deployment complete."