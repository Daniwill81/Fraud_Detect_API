import os
import json
import joblib
import numpy as np
import tensorflow as tf

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
    model = tf.keras.models.load_model(os.path.join(model_dir, 'standard_model'))
    
    # Load scaler
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    
    # Load feature names
    with open(os.path.join(model_dir, 'features.txt'), 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"Model loaded successfully. Feature names: {feature_names}")
    return model

def input_fn(request_body, request_content_type):
    """
    Parse and preprocess input data.
    
    Args:
        request_body: The request payload
        request_content_type: The content type of the request
        
    Returns:
        Preprocessed input data
    """
    if request_content_type == 'application/json':
        # Parse JSON input
        input_data = json.loads(request_body)
        data = input_data.get('data', [])
        
        # Ensure data matches expected feature count
        if len(data) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features but got {len(data)}")
        
        return np.array(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Generate predictions using the model.
    
    Args:
        input_data: Preprocessed input data
        model: The loaded model
        
    Returns:
        Model predictions
    """
    # Reshape for LSTM (batch_size, timesteps, features)
    input_data_reshaped = input_data.reshape(1, 1, -1)
    
    # Make prediction
    prediction = model.predict(input_data_reshaped)[0][0]
    
    # Calculate confidence (distance from 0.5 normalized to 0-1 range)
    confidence = abs(prediction - 0.5) * 2
    
    return {
        'prediction': float(prediction),
        'confidence': float(confidence)
    }

def output_fn(prediction, accept):
    """
    Format the prediction output.
    
    Args:
        prediction: The model's prediction
        accept: The accept content type of the response
        
    Returns:
        Formatted output
    """
    if accept == 'application/json':
        return json.dumps(prediction)
    else:
        return json.dumps(prediction)
