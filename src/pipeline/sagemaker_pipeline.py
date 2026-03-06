import os
import logging
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn import SKLearn
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") 
logger = logging.getLogger(__name__) 

def get_config(): 
    sts_client = boto3.client("sts")
	account_id = sts_client.get_caller_identity()["Account"]

	return {
		"region": "us-east-1", = "bucket": f"mlops-fraud-detection-dev-{account_id}",
		"role": f"arn:aws:iam::{account_id}:role/mlops-fraud-detection-sagemaker-role",
		"pipeline_name": "mlops-fraud-detection-pipeline",
		"framework_version": "1.2-1",
		"instance_type": "ml.m5.large",
		"input_data_uri": f"s3://mlops-fraud-detection-dev-{account_id}/raw/creditcard.csv",
	}  

def create_pipeline_parameters(config): 
    input_data = ParameterString(
    name="InputDataUrl",
    default_value=config["input_data_uri"]
)

instance_type = ParameterString(
    name="ProcessingInstanceType",
    default_value=config["instance_type"]
)

model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval"
) 

    return input_data, instance_type, model_approval_status 

def create_processing_state(config, params): 
    sklearn_processor = SKLearnProcessor(
    framework_version=config["framework_version"],
    role=config["role"],
    instance_type=params["instance_type"],
    instance_count=1,
    base_job_name="fraud-detection-preprocessing"
)

   step_process = ProcessingStep(
    name="PreprocessData",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=params["input_data"],
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
            destination=f"s3://{config['bucket']}/processed/train"
        ),
        ProcessingOutput(
            output_name="validation",
            source="/opt/ml/processing/output/validation",
            destination=f"s3://{config['bucket']}/processed/validation"
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/output/test",
            destination=f"s3://{config['bucket']}/processed/test"
        ),
        ProcessingOutput(
            output_name="scaler",
            source="/opt/ml/processing/output",
            destination=f"s3://{config['bucket']}/processed"
        )
    ],
    code="src/processing/preprocess.py",
    job_arguments=[
        "--input-data", "/opt/ml/processing/input/creditcard.csv",
        "--output-dir", "/opt/ml/processing/output"
    ]
) 
    return step_process 

def create_training_step(config, params, step_process): 
    sklearn_estimator = SKLearn(
    entry_point="train.py",
    source_dir="src/training",
    framework_version=config["framework_version"],
    role=config["role"],
    instance_type=params["instance_type"],
    instance_count=1,
    base_job_name="fraud-detection-training",
    hyperparameters={
        "max-iter": 1000,
        "random-state": 42
    },
    output_path=f"s3://{config['bucket']}/model-artifacts"
) 

    evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="metrics",
    path="metrics.json"
) 

    step_train = TrainingStep(
    name="TrainModel",
    estimator=sklearn_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
        ),
        "validation": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri
        ),
        "test": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri
        )
    },
    property_files=[evaluation_report]
) 

    return step_train, evaluation_report 