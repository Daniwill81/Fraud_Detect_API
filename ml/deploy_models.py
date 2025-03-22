import argparse
import json
import os

import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow


def deploy_models(role, bucket_name):
    """
    Deploy both standard and CKKS models to SageMaker.

    Args:
        role: AWS IAM role
        bucket_name: S3 bucket name
    """
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()

    # 1. Deploy Standard Model
    print("Deploying standard model...")

    # Upload model artifacts
    standard_model_data = sagemaker_session.upload_data(
        path="model", bucket=bucket_name, key_prefix="fraud-detection/standard-model"
    )

    # Create model
    standard_estimator = TensorFlow(
        entry_point="inference.py",
        source_dir="ml",
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="2.10",
        py_version="py39",
        model_dir=f"s3://{bucket_name}/fraud-detection/standard-model",
    )

    # Deploy model
    standard_predictor = standard_estimator.deploy(
        initial_instance_count=1, instance_type="ml.m5.large", endpoint_name="fraud-detection-standard"
    )

    print(f"Standard model deployed. Endpoint: {standard_predictor.endpoint_name}")

    # 2. Deploy CKKS Model
    print("Deploying CKKS encrypted model...")

    # Upload model artifacts
    ckks_model_data = sagemaker_session.upload_data(
        path="model_ckks", bucket=bucket_name, key_prefix="fraud-detection/ckks-model"
    )

    # Create model
    ckks_estimator = TensorFlow(
        entry_point="encrypted_inference.py",
        source_dir="ml",
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="2.10",
        py_version="py39",
        model_dir=f"s3://{bucket_name}/fraud-detection/ckks-model",
    )

    # Deploy model
    ckks_predictor = ckks_estimator.deploy(
        initial_instance_count=1, instance_type="ml.m5.large", endpoint_name="fraud-detection-ckks"
    )

    print(f"CKKS model deployed. Endpoint: {ckks_predictor.endpoint_name}")

    # Write endpoints to config file
    config = {
        "SAGEMAKER_STANDARD_ENDPOINT": standard_predictor.endpoint_name,
        "SAGEMAKER_CKKS_ENDPOINT": ckks_predictor.endpoint_name,
    }

    with open("endpoints.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Endpoints config saved to endpoints.json")

    return {"standard_endpoint": standard_predictor.endpoint_name, "ckks_endpoint": ckks_predictor.endpoint_name}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy fraud detection models to SageMaker")
    parser.add_argument("--role", type=str, required=True, help="AWS IAM role for SageMaker")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name for model artifacts")

    args = parser.parse_args()

    # Deploy models
    endpoints = deploy_models(args.role, args.bucket)

    print("\nDeployment complete!")
    print(f"Standard model endpoint: {endpoints['standard_endpoint']}")
    print(f"CKKS model endpoint: {endpoints['ckks_endpoint']}")
    print("\nUpdate these endpoints in your API configuration.")
