import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sagemaker
from sagemaker.tensorflow import TensorFlow
import argparse
import os
import boto3

def load_data(data_path):
    """
    Load and preprocess the fraud detection dataset.
    
    Args:
        data_path: Path to the CSV data file
        
    Returns:
        Processed features and target variables
    """
    # Load data from S3
    df = pd.read_csv(data_path)
    
    # Clean and prepare data
    df["Transaction Amount"].fillna(df["Transaction Amount"].median(), inplace=True)
    df["Customer Age"].fillna(df["Customer Age"].median(), inplace=True)
    
    # Add feature for address match
    df["Address_Match"] = (df["Shipping Address"] == df["Billing Address"]).astype(int)
    
    # Extract more features for better model performance
    # Example: day of week, hour of day, transaction count per customer, etc.
    if "Transaction Time" in df.columns:
        df["Transaction Time"] = pd.to_datetime(df["Transaction Time"])
        df["Hour"] = df["Transaction Time"].dt.hour
        df["Weekday"] = df["Transaction Time"].dt.weekday
    
    # Select relevant features
    features = ["Transaction Amount", "Customer Age", "Address_Match"]
    if "Hour" in df.columns:
        features.extend(["Hour", "Weekday"])
        
    # Split into features and target
    X = df[features]
    y = df["Is Fraudulent"]
    
    # Normalize feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for inference
    import joblib
    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, "model/scaler.pkl")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, features

def build_lstm_model(input_shape, lstm_units=64):
    """
    Build an LSTM model for fraud detection.
    
    Args:
        input_shape: Shape of input features
        lstm_units: Number of LSTM units
        
    Returns:
        Compiled LSTM model
    """
    # Reshape for LSTM (samples, timesteps, features)
    input_shape = (1, input_shape[0])
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(lstm_units, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(lstm_units // 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def train_model(X_train, y_train, X_test, y_test, features):
    """
    Train the LSTM model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        features: List of feature names
    """
    # Reshape data for LSTM
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
    # Build model
    model = build_lstm_model((X_train.shape[1],))
    
    # Train model
    model.fit(
        X_train_reshaped, y_train,
        validation_data=(X_test_reshaped, y_test),
        epochs=10,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test_reshaped, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save model
    os.makedirs("model", exist_ok=True)
    model.save("model/standard_model")
    
    # Save feature names
    with open("model/features.txt", "w") as f:
        f.write("\n".join(features))
    
    return model

def deploy_to_sagemaker(role, bucket_name, prefix="standard-model"):
    """
    Deploy the model to SageMaker.
    
    Args:
        role: AWS IAM role
        bucket_name: S3 bucket name
        prefix: S3 key prefix
    """
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Upload model to S3
    model_data = sagemaker_session.upload_data(
        path="model",
        bucket=bucket_name,
        key_prefix=prefix
    )
    
    # Create TensorFlow estimator
    estimator = TensorFlow(
        entry_point="inference.py",
        source_dir=".",
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="2.10",
        py_version="py39",
        model_dir=f"s3://{bucket_name}/{prefix}/model",
    )
    
    # Deploy model
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large"
    )
    
    print(f"Model deployed. Endpoint name: {predictor.endpoint_name}")
    
    return predictor.endpoint_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
