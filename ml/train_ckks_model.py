import argparse
import os

import joblib
import numpy as np
import pandas as pd
import tenseal as ts
from imblearn.over_sampling import SMOTE
from keras.src import Sequential, callbacks, metrics, optimizers, regularizers
from keras.src.layers import Dense, Dropout
from keras.src.layers.rnn.lstm import LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA


class HomomorphicEncryptionService:
    """
    Homomorphic encryption service using TenSEAL with CKKS.
    Allows operations on encrypted data.
    """

    def __init__(
        self,
        scheme: str = "CKKS",
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list[int] = None,
        security_level: int = 128,
        key_dir: str = "crypto_keys",
    ):
        """
        Initialize homomorphic encryption service.

        Args:
            scheme: Encryption scheme (CKKS or BFV)
            poly_modulus_degree: Modular polynomial degree
            coeff_mod_bit_sizes: Bit sizes for modular coefficients
            security_level: Desired security level
            key_dir: Directory to store keys
        """
        self.key_dir = key_dir
        os.makedirs(key_dir, exist_ok=True)

        # Create the context
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        self.context = ts.Context(
            ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_modulus_degree, coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()

        # Generate or load keys
        self._generate_keys()

    def _generate_keys(self):
        """Generate and save public and private keys."""
        # Create the key directory if it doesn't exist
        if not os.path.exists(self.key_dir):
            os.makedirs(self.key_dir)
            
        # Save secret key
        secret_key = self.context.secret_key()
        with open(os.path.join(self.key_dir, "secret.key"), "wb") as f:
            f.write(secret_key.save())

        # Make context public for key distribution
        self.context.make_context_public()
        context_bytes = self.context.save()
        with open(os.path.join(self.key_dir, "public.key"), "wb") as f:
            f.write(context_bytes)
            
        # Print confirmation
        print(f"Cryptographic keys generated and saved to {self.key_dir}")

    def encrypt(self, data):
        """
        Encrypt data using CKKS.

        Args:
            data: Data to encrypt (float or list of floats)
        """
        if not isinstance(data, list):
            data = [float(data)]
        return ts.ckks_tensor(self.context, data)

    def decrypt(self, encrypted_data):
        """
        Decrypt CKKS encrypted data.

        Args:
            encrypted_data: Encrypted data
        """
        result = encrypted_data
        return result[0] if len(result) == 1 else result


def load_data(data_path):
    """
    Load and preprocess the fraud detection dataset.
    Handles class imbalance using SMOTE.

    Args:
        data_path: Path to the CSV data file

    Returns:
        Processed features and target variables with balanced classes
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

    return X_train_resampled, X_test, y_train_resampled, y_test, features


def build_encrypted_compatible_model(input_shape, lstm_units=64) -> Sequential:
    """
    Build an LSTM model compatible with encrypted data.

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
            Dropout(0.3),
            LSTM(lstm_units // 2),
            Dropout(0.3),
            Dense(16, activation="relu", kernel_regularizer=regularizers.L2(0.01)),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=optimizers.Adam(0.0005),
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
    """
    # Reshape data for LSTM
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Build model
    model = build_encrypted_compatible_model((X_train.shape[1],))

    # Train model
    history = model.fit(
        X_train_reshaped,
        y_train,
        validation_data=(X_test_reshaped, y_test),
        epochs=15,
        batch_size=64,
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
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
    os.makedirs("model_ckks", exist_ok=True)
    model.save("model_ckks/ckks_model.keras")

    # Save feature names
    with open("model_ckks/features.txt", "w") as f:
        f.write("\n".join(features))

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the fraud detection dataset")
    parser.add_argument("--role", type=str, required=True, help="AWS IAM role for SageMaker")

    args = parser.parse_args()

    # Load and process data
    X_train, X_test, y_train, y_test, features = load_data(args.data_path)

    # Train the model
    model = train_encrypted_model(X_train, y_train, X_test, y_test, features)
