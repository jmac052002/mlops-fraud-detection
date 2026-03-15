import json 
import logging
import boto3
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker_client = boto3.client('sagemaker')

def handler(event, context):
# Log the incoming event: logger.info(f"Received event: {json.dumps(event)}")
    # Get the pipeline name from an environment variable: pipeline_name = os.environ["PIPELINE_NAME"]
    # Get the S3 bucket and key from the event (extract this from the EventBridge event format)
    # Build the input data URI from the bucket and key 
    # Start the SageMaker pipeline execution with the input data URI as a parameter
    logger.info(f"Received event: {json.dumps(event)}")
    pipeline_name = os.environ["PIPELINE_NAME"]
    # Extract bucket and key from the event (assuming it's an EventBridge event with S3 details)
    bucket = event['detail']['bucket']['name']
    key = event['detail']['object']['key']
    input_data_url = f"s3://{bucket}/{key}"
    
    
    if not key.startswith("raw/") or not key.endswith(".csv"):
        logger.info(f"Skipping non-data file: {key}")
        return {"statusCode": 200, "body": "Skipped - not a raw CSV file"}
    
    
    logger.info(f"Starting SageMaker pipeline '{pipeline_name}' with input data URL: {input_data_url}")
    response = sagemaker_client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName=f"Execution for {key}",
        PipelineParameters=[
            {
                'Name': 'InputDataUrl',
                'Value': input_data_url
            }
        ]
    )
    logger.info(f"SageMaker pipeline execution started: {response['PipelineExecutionArn']}") 
    return {
        'statusCode': 200,
        'body': json.dumps(f"SageMaker pipeline execution started: {response['PipelineExecutionArn']}")
    }

