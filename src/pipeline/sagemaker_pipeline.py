import json
import os
import logging
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn import SKLearn
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# This function returns a dictionary with all the values specific to my AWS environment.
# Keeping them in one place makes the script easy to update and
# reuse across different environments (dev, staging, prod) by just changing the values in this function.
def get_config():
    sts_client = boto3.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]

    return {
        "region": "us-east-1",
        "bucket": f"mlops-fraud-detection-dev-{account_id}",
        "role": f"arn:aws:iam::{account_id}:role/mlops-fraud-detection-sagemaker-role",
        "pipeline_name": "mlops-fraud-detection-pipeline",
        "framework_version": "1.2-1",
        "instance_type": "ml.m5.large",
        "input_data_uri": f"s3://mlops-fraud-detection-dev-{account_id}/raw/creditcard.csv",
    }


# This function creates the SageMaker Pipeline parameters that will be used to make the pipeline flexible and reusable.
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

    f1_threshold = ParameterFloat(
        name="F1Threshold",
        default_value=0.10
    )
    return input_data, instance_type, model_approval_status, f1_threshold


def create_processing_step(config, params, pipeline_session):
    # Define the SKLearnProcessor which will run the preprocessing script in a SageMaker Processing Job.
    sklearn_processor = SKLearnProcessor(
        framework_version=config["framework_version"],
        role=config["role"],
        instance_type=params["instance_type"],
        instance_count=1,
        base_job_name="fraud-detection-preprocessing",
        sagemaker_session=pipeline_session
    )

    # Define the processing step with inputs, outputs, and the preprocessing script that will run the data preprocessing logic.
    # The script will read the raw data from S3, preprocess it, and save the processed train/validation/test sets back to S3 for the next step.
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


# This function creates the training step of the pipeline, which will train a machine learning model
# using the preprocessed data from the previous step.
def create_training_step(config, params, step_process, pipeline_session):
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
        output_path=f"s3://{config['bucket']}/model-artifacts",
        sagemaker_session=pipeline_session
    )

    # Define the training step with the estimator, inputs from the processing step, and the property file to capture the evaluation metrics.
    # The training script will read the preprocessed train/validation/test data from S3, train the model
    # and save the evaluation metrics to the specified property file for later use in the condition step.
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
        }
    )

    return step_train


def create_evaluation_step(config, params, step_process, step_train, pipeline_session):
    # Create a processor to run the evaluation script
    eval_processor = SKLearnProcessor(
        framework_version=config["framework_version"],
        role=config["role"],
        instance_type=params["instance_type"],
        instance_count=1,
        base_job_name="fraud-detection-evaluation",
        sagemaker_session=pipeline_session
    )

    # PropertyFile tells SageMaker where to find the metrics JSON output
    # so the condition step can read values from it
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="metrics",
        path="metrics.json"
    )

    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="metrics",
                source="/opt/ml/processing/output",
                destination=f"s3://{config['bucket']}/evaluation"
            )
        ],
        code="src/evaluation/evaluate.py",
        job_arguments=[
            "--model-dir", "/opt/ml/processing/model",
            "--test-data-dir", "/opt/ml/processing/test",
            "--output-dir", "/opt/ml/processing/output"
        ],
        property_files=[evaluation_report]
    )

    return step_evaluate, evaluation_report


def create_condition_step(params, step_evaluate, evaluation_report, step_register):
    # Quality gate: only register the model if F1 score meets the threshold
    condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="f1_score"
        ),
        right=params["f1_threshold"]
    )

    step_condition = ConditionStep(
        name="CheckModelQuality",
        conditions=[condition],
        if_steps=[step_register],
        else_steps=[]
    )

    return step_condition


def create_register_step(config, params, step_train, pipeline_session):
    # Register the trained model in SageMaker Model Registry
    model = Model(
        image_uri=sagemaker.image_uris.retrieve(
            "sklearn",
            config["region"],
            version=config["framework_version"]
        ),
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=config["role"]
    )

    step_register = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name="fraud-detection-model-group",
            approval_status=params["model_approval_status"]
        )
    )

    return step_register

# This function creates the SageMaker Pipeline by combining all the steps (processing, training, condition)
# and defining the pipeline parameters.
def create_pipeline(config, params, step_process, step_train, step_evaluate, step_condition, pipeline_session):
    pipeline = Pipeline(
        name=config["pipeline_name"],
        parameters=[
            params["input_data"],
            params["instance_type"],
            params["model_approval_status"],
            params["f1_threshold"]
        ],
        steps=[step_process, step_train, step_evaluate, step_condition],
        sagemaker_session=pipeline_session
    )

    return pipeline

# The main function orchestrates the entire process of building, upserting, and executing the SageMaker Pipeline.
def main():
    logger.info("Building SageMaker Pipeline...")

    # PipelineSession defers all SageMaker API calls until pipeline execution
    # This is what allows pipeline variables to be used without immediate resolution
    config = get_config()

    pipeline_session = PipelineSession()

    # Create parameters
    params_dict = create_pipeline_parameters(config)
    params = {
        "input_data": params_dict[0],
        "instance_type": params_dict[1],
        "model_approval_status": params_dict[2],
        "f1_threshold": params_dict[3]
    }

   # Create steps
    step_process = create_processing_step(config, params, pipeline_session)
    step_train = create_training_step(config, params, step_process, pipeline_session)
    step_evaluate, evaluation_report = create_evaluation_step(config, params, step_process, step_train, pipeline_session)
    step_register = create_register_step(config, params, step_train, pipeline_session)
    step_condition = create_condition_step(params, step_evaluate, evaluation_report, step_register)

    # Build pipeline
    pipeline = create_pipeline(config, params, step_process, step_train, step_evaluate, step_condition, pipeline_session)
    
    # Submit to SageMaker
    logger.info("Upserting pipeline...")
    pipeline.upsert(role_arn=config["role"])

    # Start execution
    logger.info("Starting pipeline execution...")
    execution = pipeline.start()
    logger.info(f"Pipeline execution started: {execution.arn}")

    # Wait for completion
    logger.info("Waiting for pipeline to complete...")
    execution.wait()
    logger.info("Pipeline execution complete!")

    # Get final execution status
    execution_status = execution.describe()['PipelineExecutionStatus']
    logger.info(f"Final status: {execution_status}")

    # Invoke monitoring Lambda to send alerts and publish CloudWatch metrics
    lambda_client = boto3.client('lambda')
    lambda_client.invoke(
        FunctionName='mlops-fraud-detection-monitor',
        InvocationType='Event',
        Payload=json.dumps({
            'detail': {
                'pipelineExecutionArn': execution.arn,
                'currentPipelineExecutionStatus': execution_status
            }
        })
    )
    logger.info("Monitoring Lambda invoked successfully")


if __name__ == "__main__":
    main()
