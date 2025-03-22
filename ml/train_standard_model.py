import argparse
import os

import boto3
import joblib
import numpy as np
import pandas as pd
import sagemaker
from imblearn.over_sampling import SMOTE
from keras.src import Sequential, callbacks, metrics, optimizers
from keras.src.layers import Dense, Dropout
from keras.src.layers.rnn.lstm import LSTM
from sagemaker.tensorflow import TensorFlow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(data_path):
    """
    Load and preprocess the fraud detection dataset.
    Handles class imbalance using SMOTE.

    Args:
        data_path: Path to the CSV data file

    Returns:
        Processed features and target variables with balanced classes
    """
    # Load data from S3
    df = pd.read_csv(data_path)

    # Clean and prepare data
    df["Transaction Amount"].fillna(df["Transaction Amount"].median(), inplace=True)
    df["Customer Age"].fillna(df["Customer Age"].median(), inplace=True)

    # Add feature for address match
    df["Address_Match"] = (df["Shipping Address"] == df["Billing Address"]).astype(int)

    # Extract more features for better model performance
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
    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, "model/scaler.pkl")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE to handle imbalanced data (only on training data)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Check resampled distribution
    print(f"Class distribution after balancing: {pd.Series(y_train_resampled).value_counts(normalize=True) * 100}")

    return X_train_resampled, X_test, y_train_resampled, y_test, features


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

    model = Sequential(
        [
            LSTM(lstm_units, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(lstm_units // 2),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    # Use class weights for additional balance during training
    model.compile(
        optimizer=optimizers.Adam(0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc"),
        ],
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
        X_train_reshaped,
        y_train,
        validation_data=(X_test_reshaped, y_test),
        epochs=10,
        batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
        ],
    )

    # Evaluate model
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

    # Save model
    os.makedirs("model", exist_ok=True)
    model.save("model/balanced_model")

    # Save feature names
    with open("model/features.txt", "w") as f:
        f.write("\n".join(features))

    return model


def deploy_to_sagemaker(role, bucket_name, prefix="balanced-model"):
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
    model_data = sagemaker_session.upload_data(path="model", bucket=bucket_name, key_prefix=prefix)

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
    predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.m5.large")

    print(f"Model deployed. Endpoint name: {predictor.endpoint_name}")

    return predictor.endpoint_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the fraud detection dataset")
    parser.add_argument("--role", type=str, required=True, help="AWS IAM role for SageMaker")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name for storing model artifacts")
    parser.add_argument("--deploy", action="store_true", help="Whether to deploy model to SageMaker")

    args = parser.parse_args()

    # Load and process data
    X_train, X_test, y_train, y_test, features = load_data(args.data_path)

    # Train the model
    model = train_model(X_train, y_train, X_test, y_test, features)

    # Deploy if requested
    if args.deploy:
        endpoint_name = deploy_to_sagemaker(args.role, args.bucket)
        print(f"Model deployed to endpoint: {endpoint_name}")
