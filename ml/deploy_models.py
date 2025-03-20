import argparse
import os
import boto3
import json
import sagemaker
from sagemaker.tensorflow import TensorFlow

def create_inference_script(script_path, is_encrypted=False):
    """
    Create an inference script for the model.
    
    Args:
        script_path: Path to save the script
        is_encrypted: Whether this is for the encrypted model
    """
    # Create inference script for standard model
    if not is_encrypted:
        script_content = """
import os
import json
import joblib
import numpy as np
import tensorflow as tf

# Load the model and scaler
model = None
scaler = None

def model_fn(model_dir):
    global model, scaler
    model = tf.keras.models.load_model(os.path.join(model_dir, 'standard_model'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    
    # Load feature names
    with open(os.path.join(model_dir, 'features.txt'), 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        data = input_data.get('data', [])
        return np.array(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    # Reshape for LSTM (assuming input_data is a vector of features)
    input_data_reshaped = input_data.reshape(1, 1, -1)
    
    # Get prediction
    prediction = model.predict(input_data_reshaped)[0][0]
    
    # Calculate confidence (distance from 0.5)
    confidence = abs(prediction - 0.5) * 2
    
    return {
        'prediction': float(prediction),
        'confidence': float(confidence)
    }

def output_fn(prediction, accept):
    return json.dumps(prediction)
"""
    else:
        # Create inference script for encrypted model
        script_content = """
import os
import json
import joblib
import numpy as np
import tensorflow as tf

# Load the model, scaler, and CKKS simulator
model = None
scaler = None
ckks_simulator = None

def model_fn(model_dir):
    global model, scaler, ckks_simulator
    model = tf.keras.models.load_model(os.path.join(model_dir, 'ckks_model'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    ckks_simulator = joblib.load(os.path.join(model_dir, 'ckks_simulator.pkl'))
    
    # Load feature names
    with open(os.path.join(model_dir, 'features.txt'), 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        data = input_data.get('data', [])
        is_encrypted = input_data.get('encrypted', False)
        
        if is_encrypted:
            # Data is already encrypted, we just need to decrypt it for the model
            # In a real HE system, this would happen securely
            decrypted_data = ckks_simulator.decrypt_vector(data)
            return np.array(decrypted_data)
        else:
            # Raw data - normally we wouldn't handle this in the encrypted endpoint
            # but including for testing/comparison
            return np.array(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    # Reshape for LSTM (assuming input_data is a vector of features)
    input_data_reshaped = input_data.reshape(1, 1, -1)
    
    # Get prediction
    prediction = model.predict(input_data_reshaped)[0][0]
    
    # Calculate confidence (distance from 0.5)
    confidence = abs(prediction - 0.5) * 2
    
    return {
        'prediction': float(prediction),
        'confidence': float(confidence)
    }

def output_fn(prediction, accept):
    return json.dumps(prediction)
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)

def deploy_models(role, bucket_name):
    """
    Deploy both standard and CKKS models to SageMaker.
    
    Args:
        role: AWS IAM role
        bucket_name: S3 bucket name
    """
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Create inference scripts
    os.makedirs("deployment", exist_ok=True)
    create_inference_script("deployment/inference.py", is_encrypted=False)
    create_inference_script("deployment/encrypted_inference.py", is_encrypted=True)
    
    # 1. Deploy Standard Model
    print("Deploying standard model...")
    
    # Upload model artifacts
    standard_model_data = sagemaker_session.upload_data(
        path="model",
        bucket=bucket_name,
        key_prefix="fraud-detection/standard-model"
    )
    
    # Create model
    standard_estimator = TensorFlow(
        entry_point="inference.py",
        source_dir="deployment",
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="2.10",
        py_version="py39",
        model_dir=f"s3://{bucket_name}/fraud-detection/standard-model"
    )
    
    # Deploy model
    standard_predictor = standard_estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name="fraud-detection-standard"
    )
    
    print(f"Standard model deployed. Endpoint: {standard_predictor.endpoint_name}")
    
    # 2. Deploy CKKS Model
    print("Deploying CKKS encrypted model...")
    
    # Upload model artifacts
    ckks_model_data = sagemaker_session.upload_data(
        path="model_ckks",
        bucket=bucket_name,
        key_prefix="fraud-detection/ckks-model"
    )
    
    # Create model
    ckks_estimator = TensorFlow(
        entry_point="encrypted_inference.py",
        source_dir="deployment",
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="2.10",
        py_version="py39",
        model_dir=f"s3://{bucket_name}/fraud-detection/ckks-model"
    )
    
    # Deploy model
    ckks_predictor = ckks_estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name="fraud-detection-ckks"
    )
    
    print(f"CKKS model deployed. Endpoint: {ckks_predictor.endpoint_name}")
    
    # Write endpoints to config file
    config = {
        "SAGEMAKER_STANDARD_ENDPOINT": standard_predictor.endpoint_name,
        "SAGEMAKER_CKKS_ENDPOINT": ckks_predictor.endpoint_name
    }
    
    with open("deployment/endpoints.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Endpoints config saved to deployment/endpoints.json")
    
    return {
        "standard_endpoint": standard_predictor.endpoint_name,
        "ckks_endpoint": ckks_predictor.endpoint_name
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy fraud detection models to SageMaker")
    parser.add_argument("--role", type=str, required=True, 
                        help="AWS IAM role for SageMaker")
    parser.add_argument("--bucket", type=str, required=True, 
                        help="S3 bucket name for model artifacts")
    
    args = parser.parse_args()
    
    # Deploy models
    endpoints = deploy_models(args.role, args.bucket)
    
    print("\nDeployment complete!")
    print(f"Standard model endpoint: {endpoints['standard_endpoint']}")
    print(f"CKKS model endpoint: {endpoints['ckks_endpoint']}")
    print("\nUpdate these endpoints in your API configuration.")
