import json
import logging
import os

import joblib
import numpy as np
from tensorflow.python.keras import models

from utils.encryption import decrypt_data, encrypt_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for the model and related components
model = None
scaler = None
feature_names = None


def model_fn(model_dir):
    """
    Load the model artifacts from disk.
    This function is called by SageMaker when the endpoint starts.

    Args:
        model_dir: Directory containing model artifacts

    Returns:
        The loaded model
    """
    global model, scaler, feature_names

    # Load TensorFlow model
    logger.info(f"Loading model from {model_dir}")
    model = models.load_model(os.path.join(model_dir, "ckks_model"))

    # Load scaler
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    # Load feature names
    with open(os.path.join(model_dir, "features.txt"), "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

    logger.info(f"Model loaded successfully. Feature names: {feature_names}")
    return model


def input_fn(request_body, request_content_type):
    """
    Parse and preprocess encrypted input data.

    Args:
        request_body: The request payload
        request_content_type: The content type of the request

    Returns:
        Preprocessed input data
    """
    if request_content_type == "application/json":
        # Parse JSON input
        input_data = json.loads(request_body)
        encrypted_data = input_data.get("data", [])
        is_encrypted = input_data.get("encrypted", True)

        logger.info(f"Received {'encrypted' if is_encrypted else 'raw'} data")

        if is_encrypted:
            # In a real system, this would decrypt within the secure environment
            try:
                decrypted_data = decrypt_data(encrypted_data)
                logger.info("Data decrypted successfully")
                return np.array(decrypted_data)
            except Exception as e:
                logger.error(f"Error decrypting data: {str(e)}")
                raise
        else:
            # For testing: process raw data directly
            logger.info("Processing raw data directly")
            return np.array(encrypted_data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Generate predictions using the model on encrypted/decrypted data.

    Args:
        input_data: Preprocessed input data
        model: The loaded model

    Returns:
        Model predictions
    """
    try:
        # Reshape for LSTM (batch_size, timesteps, features)
        input_data_reshaped = input_data.reshape(1, 1, -1)

        # Make prediction
        logger.info("Making prediction")
        prediction = model.predict(input_data_reshaped)[0][0]

        # Calculate confidence (distance from 0.5 normalized to 0-1 range)
        confidence = abs(prediction - 0.5) * 2

        logger.info(f"Prediction: {prediction}, Confidence: {confidence}")

        return {"prediction": float(prediction), "confidence": float(confidence)}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


def output_fn(prediction, accept):
    """
    Format the prediction output.

    Args:
        prediction: The model's prediction
        accept: The accept content type of the response

    Returns:
        Formatted output
    """
    if accept == "application/json":
        return json.dumps(prediction)
    else:
        return json.dumps(prediction)
