import boto3
import logging
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
REGION = "us-east-1"
ROLE = "arn:aws:iam::584996267165:role/mlops-fraud-detection-sagemaker-role"
MODEL_ARTIFACTS = "s3://mlops-fraud-detection-dev-584996267165/model-artifacts/pipelines-ufue6z6nei99-TrainModel-E5euK5gy8J/output/model.tar.gz"
ENDPOINT_NAME = "fraud-detection-endpoint"
INSTANCE_TYPE = "ml.m5.large"
FRAMEWORK_VERSION = "1.2-1"

def deploy_endpoint(model_artifacts, endpoint_name, instance_type, role):
    """Deploy sklearn model directly to a SageMaker endpoint"""
    sagemaker_session = sagemaker.Session()

    model = SKLearnModel(
        model_data=model_artifacts,
        role=role,
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=sagemaker_session,
        entry_point="inference.py",
        source_dir="src/deployment"
    )

    logger.info(f"Deploying model to endpoint: {endpoint_name}")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    logger.info(f"Endpoint deployed successfully: {endpoint_name}")
    return predictor

def main():
    logger.info("Starting model deployment...")
    predictor = deploy_endpoint(
        model_artifacts=MODEL_ARTIFACTS,
        endpoint_name=ENDPOINT_NAME,
        instance_type=INSTANCE_TYPE,
        role=ROLE
    )
    logger.info(f"Deployment complete! Endpoint: {ENDPOINT_NAME}")
    logger.info("Remember to delete the endpoint when done to avoid charges:")
    logger.info(f"aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME}")

if __name__ == "__main__":
    main() 