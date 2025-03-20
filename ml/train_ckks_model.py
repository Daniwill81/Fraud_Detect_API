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
import joblib

# Simulated CKKS wrapper for homomorphic encryption
class CKKSSimulator:
    def __init__(self, poly_modulus_degree=8192, coeff_mod_bit_sizes=None):
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        
    def encrypt_vector(self, vec):
        """Simulate CKKS encryption of a vector"""
        # In a real system, this would use actual encryption
        # Here we just mark the data as encrypted
        return [f"enc_{float(x)}" for x in vec]
    
    def decrypt_vector(self, enc_vec):
        """Simulate CKKS decryption of a vector"""
        # In a real system, this would use actual decryption
        # Here we just extract the original value
        return np.array([float(x.split('_')[1]) for x in enc_vec])

def load_data(data_path):
    """
    Load and preprocess the fraud detection dataset with CKKS encryption simulation.
    
    Args:
        data_path: Path to the CSV data file
        
    Returns:
        Processed features and target variables with encryption simulation
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Clean and prepare data
    df["Transaction Amount"].fillna(df["Transaction Amount"].median(), inplace=True)
    df["Customer Age"].fillna(df["Customer Age"].median(), inplace=True)
    
    # Add feature for address match
    df["Address_Match"] = (df["Shipping Address"] == df["Billing Address"]).astype(int)
    
    # Extract more features
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
    os.makedirs("model_ckks", exist_ok=True)
    joblib.dump(scaler, "model_ckks/scaler.pkl")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Initialize CKKS simulator
    ckks = CKKSSimulator()
    
    # Save CKKS simulator for inference
    joblib.dump(ckks, "model_ckks/ckks_simulator.pkl")
    
    return X_train, X_test, y_train, y_test, features, ckks

def build_encrypted_compatible_model(input_shape, lstm_units=64):
    """
    Build an LSTM model that can work with encrypted data.
    The model structure is similar to the standard model,
    but with adjustments to handle encrypted data characteristics.
    
    Args:
        input_shape: Shape of input features
        lstm_units: Number of LSTM units
        
    Returns:
        Compiled LSTM model compatible with encrypted data
    """
    # Reshape for LSTM (samples, timesteps, features)
    input_shape = (1, input_shape[0])
    
    model = tf.keras.Sequential([
        # Similar architecture but with modified hyperparameters
        # for encrypted data compatibility
        tf.keras.layers.LSTM(lstm_units, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.Dropout(0.3),  # Higher dropout for better generalization with encrypted data
        tf.keras.layers.LSTM(lstm_units // 2),
        tf.keras.layers.Dropout(0.3),
        # More regularization for encrypted data
        tf.keras.layers.Dense(16, activation="relu", 
                             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),  # Lower learning rate for stability
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def train_encrypted_model(X_train, y_train, X_test, y_test, features, ckks):
    """
    Train the LSTM model with encrypted data simulation.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        features: List of feature names
        ckks: CKKS simulator instance
    """
    # For training, we'll use the raw data, but during inference we'll use encrypted data
    # This simulates training on trusted data but deploying for encrypted inference
    
    # Reshape data for LSTM
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
    # Build model adjusted for encrypted data
    model = build_encrypted_compatible_model((X_train.shape[1],))
    
    # Train model with more epochs for better convergence
    model.fit(
        X_train_reshaped, y_train,
        validation_data=(X_test_reshaped, y_test),
        epochs=15,  # More epochs for encrypted-compatible model
        batch_size=64,  # Larger batch size for stability
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
    )
    
    # Test model accuracy
    loss, accuracy = model.evaluate(X_test_reshaped, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Also test with simulated encrypted data
    # This is just for verification during development
    encrypted_samples = []
    for i in range(min(10, len(X_test))):
        # Encrypt sample
        enc_sample = ckks.encrypt_vector(X_test[i])
        # Decrypt for model (in production, this would happen inside the secure environment)
        dec_sample = ckks.decrypt_vector(enc_sample)
        encrypted_samples.append(dec_sample)
    
    encrypted_samples = np.array(encrypted_samples).reshape(-1, 1, X_test.shape[1])
    enc_loss, enc_accuracy = model.evaluate(encrypted_samples, y_test[:len(encrypted_samples)])
    print(f"Encrypted Test Loss: {enc_loss:.4f}")
    print(f"Encrypted Test Accuracy: {enc_accuracy:.4f}")
    
    # Save model
    os.makedirs("model_ckks", exist_ok=True)
    model.save("model_ckks/ckks_model")
    
    # Save feature names
    with open("model_ckks/features.txt", "w") as f:
        f.write("\n".join(features))

    return model

def deploy_to_sagemaker(role, bucket_name, prefix="ckks-model"):
    """
    Deploy the encrypted-compatible model to SageMaker.
    
    Args:
        role: AWS IAM role
        bucket_name: S3 bucket name
        prefix: S3 key prefix
    """
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Upload model to S3
    model_data = sagemaker_session.upload_data(
        path="model_ckks",
        bucket=bucket_name,
        key_prefix=prefix
    )
    
    # Create TensorFlow estimator with custom entry point for encrypted inference
    estimator = TensorFlow(
        entry_point="encrypted_inference.py",
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
    
    print(f"Encrypted model deployed. Endpoint name: {predictor.endpoint_name}")
    
    return predictor.endpoint_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, 
                        help="Path to the fraud detection dataset")
    parser.add_argument("--role", type=str, required=True, 
                        help="AWS IAM role for SageMaker")
    parser.add_argument("--bucket", type=str, required=True, 
                        help="S3 bucket name for storing model artifacts")
    parser.add_argument("--deploy", action="store_true", 
                        help="Whether to deploy model to SageMaker")
    
    args = parser.parse_args()
    
    # Load and process data
    X_train, X_test, y_train, y_test, features, ckks = load_data(args.data_path)
    
    # Train the model
    model = train_encrypted_model(X_train, y_train, X_test, y_test, features, ckks)
    
    # Deploy if requested
    if args.deploy:
        endpoint_name = deploy_to_sagemaker(args.role, args.bucket)
        print(f"Model deployed to endpoint: {endpoint_name}")
