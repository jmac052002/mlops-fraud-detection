import boto3
import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
REGION = "us-east-1"
ROLE = "arn:aws:iam::584996267165:role/mlops-fraud-detection-sagemaker-role"
BUCKET = "mlops-fraud-detection-dev-584996267165"
ENDPOINT_NAME = "fraud-detection-endpoint"
BASELINE_DATA = f"s3://{BUCKET}/raw/creditcard.csv"
BASELINE_OUTPUT = f"s3://{BUCKET}/model-monitor/baseline"
MONITOR_OUTPUT = f"s3://{BUCKET}/model-monitor/reports"

def create_baseline(role, baseline_data, baseline_output):
    """Run a baseline job to establish statistics and constraints"""
    monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600
    )

    logger.info("Starting baseline job - this takes 5-10 minutes...")
    monitor.suggest_baseline(
        baseline_dataset=baseline_data,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=baseline_output,
        wait=True,
        logs=False
    )
    logger.info(f"Baseline complete. Results at: {baseline_output}")
    return monitor

def create_monitoring_schedule(monitor, endpoint_name, monitor_output):
    """Create an hourly monitoring schedule on the endpoint"""
    from sagemaker.model_monitor import CronExpressionGenerator

    logger.info(f"Creating monitoring schedule for endpoint: {endpoint_name}")
    monitor.create_monitoring_schedule(
        monitor_schedule_name="fraud-detection-monitor-schedule",
        endpoint_input=endpoint_name,
        output_s3_uri=monitor_output,
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True
    )
    logger.info("Monitoring schedule created - runs every hour")

def main():
    logger.info("Setting up monitoring schedule (baseline already complete)...")

    sagemaker_session = sagemaker.Session()

    # Load existing baseline results
    monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=1,
        instance_type="ml.m5.large",
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=sagemaker_session
    )

    from sagemaker.model_monitor import CronExpressionGenerator
    from sagemaker.model_monitor import Statistics, Constraints

    statistics = Statistics.from_s3_uri(f"{BASELINE_OUTPUT}/statistics.json")
    constraints = Constraints.from_s3_uri(f"{BASELINE_OUTPUT}/constraints.json")

    logger.info(f"Creating monitoring schedule for endpoint: {ENDPOINT_NAME}")
    monitor.create_monitoring_schedule(
        monitor_schedule_name="fraud-detection-monitor-schedule",
        endpoint_input=ENDPOINT_NAME,
        output_s3_uri=MONITOR_OUTPUT,
        statistics=statistics,
        constraints=constraints,
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True
    )
    logger.info("Monitoring schedule created successfully!")
    logger.info("Monitor runs every hour and compares incoming data to baseline")

if __name__ == "__main__":
    main()
