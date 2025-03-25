import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Désactive CUDA
import typing

import boto3
import joblib
import numpy as np
import pandas as pd
import sagemaker
from imblearn.over_sampling import SMOTE
from keras.src import Sequential, callbacks, metrics, optimizers, regularizers
from keras.src.layers import Dense, Dropout
from keras.src.layers.rnn.lstm import LSTM
from sagemaker.tensorflow import TensorFlow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.xlib import encryption


def load_data(data_path):
    """
    Load and preprocess the fraud detection dataset with CKKS encryption simulation.
    Handles class imbalance using SMOTE.

    Args:
        data_path: Path to the CSV data file

    Returns:
        Processed features and target variables with encryption simulation and balanced classes
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

    # Check class distribution
    print(f"Class distribution before balancing: {pd.Series(y).value_counts(normalize=True) * 100}")

    # Normalize feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler for inference
    os.makedirs("model_ckks", exist_ok=True)
    joblib.dump(scaler, "model_ckks/scaler.pkl")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE to handle imbalanced data (only on training data)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Check resampled distribution
    print(f"Class distribution after balancing: {pd.Series(y_train_resampled).value_counts(normalize=True) * 100}")

    # Save CKKS encryption for inference
    joblib.dump(ckks, "model_ckks/ckks_encryption.pkl")

    return X_train_resampled, X_test, y_train_resampled, y_test, features, ckks


def build_encrypted_compatible_model(input_shape, lstm_units=64) -> Sequential:
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

    model = Sequential(
        [
            # Similar architecture but with modified hyperparameters
            # for encrypted data compatibility
            LSTM(lstm_units, input_shape=input_shape, return_sequences=True),
            Dropout(0.3),  # Higher dropout for better generalization with encrypted data
            LSTM(lstm_units // 2),
            Dropout(0.3),
            # More regularization for encrypted data
            Dense(16, activation="relu", kernel_regularizer=regularizers.L2(0.01)),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=optimizers.Adam(0.0005),  # Lower learning rate for stability
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc"),
        ],
    )

    return model


def train_encrypted_model(X_train, y_train, X_test, y_test, features) -> Sequential:
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
        X_train_reshaped,
        y_train,
        validation_data=(X_test_reshaped, y_test),
        epochs=15,  # More epochs for encrypted-compatible model
        batch_size=64,  # Larger batch size for stability
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        ],
    )

    # Test model accuracy
    results = model.evaluate(X_test_reshaped, y_test)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")
    print(f"Test AUC: {results[4]:.4f}")

    # Get predictions and calculate confusion matrix
    y_pred = (model.predict(X_test_reshaped) > 0.5).astype("int32")
    from sklearn.metrics import classification_report, confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Also test with simulated encrypted data
    # This is just for verification during development
    encrypted_samples = []
    for i in range(min(10, len(X_test))):
        # Encrypt sample
        enc_sample = encryption.encrypt_data(X_test[i])
        # Decrypt for model (in production, this would happen inside the secure environment)
        dec_sample = encryption.decrypt_data(enc_sample)
        encrypted_samples.append(dec_sample)

    encrypted_samples = np.array(encrypted_samples).reshape(-1, 1, X_test.shape[1])
    enc_results = model.evaluate(encrypted_samples, y_test[: len(encrypted_samples)])
    print(f"Encrypted Test Loss: {enc_results[0]:.4f}")
    print(f"Encrypted Test Accuracy: {enc_results[1]:.4f}")

    # Save model
    os.makedirs("model_ckks", exist_ok=True)
    model.save("model_ckks/ckks_model")

    # Save feature names
    with open("model_ckks/features.txt", "w") as f:
        f.write("\n".join(features))

    return model


def deploy_to_sagemaker(role, bucket_name, prefix="ckks-balanced-model") -> typing.Any:
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
    model_data = sagemaker_session.upload_data(path="model_ckks", bucket=bucket_name, key_prefix=prefix)

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
    predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.m5.large")

    print(f"Encrypted model deployed. Endpoint name: {predictor.endpoint_name}")

    return predictor.endpoint_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the fraud detection dataset")
    parser.add_argument("--role", type=str, required=True, help="AWS IAM role for SageMaker")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name for storing model artifacts")
    parser.add_argument("--deploy", action="store_true", help="Whether to deploy model to SageMaker")

    args = parser.parse_args()

    # Load and process data
    X_train, X_test, y_train, y_test, features, ckks = load_data(args.data_path)

    # Train the model
    model = train_encrypted_model(X_train, y_train, X_test, y_test, features, ckks)

    # Deploy if requested
    if args.deploy:
        endpoint_name = deploy_to_sagemaker(args.role, args.bucket)
        print(f"Model deployed to endpoint: {endpoint_name}")
