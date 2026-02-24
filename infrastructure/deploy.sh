#!/bin/bash
set -euo pipefail

STACK_NAME="mlops-fraud-detection-data-layer"
TEMPLATE="infrastructure/data-layer.yaml"

if [[ -x ".venv/bin/cfn-lint" ]]; then
  CFN_LINT_BIN=".venv/bin/cfn-lint"
elif command -v cfn-lint >/dev/null 2>&1; then
  CFN_LINT_BIN="$(command -v cfn-lint)"
else
  echo "[cfn-lint] cfn-lint not found. Install dependencies with: pip install -r requirements.txt"
  echo "More help: https://github.com/aws-cloudformation/cfn-python-lint/#install"
  exit 1
fi

echo "Running cfn-lint on ${TEMPLATE}..."
"$CFN_LINT_BIN" "$TEMPLATE"

aws cloudformation deploy \
  --template-file "$TEMPLATE" \
  --stack-name "$STACK_NAME" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProjectName=mlops-fraud-detection \
    Environment=dev