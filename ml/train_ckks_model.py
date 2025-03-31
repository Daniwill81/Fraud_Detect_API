# train_ckks_model file
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
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=None,
        encryption_type=ts.ENCRYPTION_TYPE.SYMMETRIC,
        key_dir="crypto_keys",
    ):
        """
        Initialize homomorphic encryption service.

        Args:
            scheme: Encryption scheme (CKKS or BFV)
            poly_modulus_degree: Modular polynomial degree
            coeff_mod_bit_sizes: Bit sizes for modular coefficients
            encryption_type: Type of encryption (SYMMETRIC or ASYMMETRIC)
            key_dir: Directory to store keys
        """
        self.key_dir = key_dir
        os.makedirs(key_dir, exist_ok=True)

        # Create the context
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        self.context = ts.Context(
            scheme=scheme,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            encryption_type=encryption_type,
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()

        # Generate or load keys
        self._generate_keys()

    def _generate_keys(self):
        """Generate and save public and private keys."""
        # Create the key directory if it doesn't exist
        if not os.path.exists(self.key_dir):
            os.makedirs(self.key_dir)

        # Save all keys in one file (private context)
        private_context_data = self.context.serialize(
            save_public_key=True, save_secret_key=True, save_galois_keys=True, save_relin_keys=True
        )

        with open(os.path.join(self.key_dir, "private_context.bin"), "wb") as f:
            f.write(private_context_data)

        # Save public context (without secret key)
        public_context_data = self.context.serialize(
            save_public_key=True, save_secret_key=False, save_galois_keys=True, save_relin_keys=True
        )

        with open(os.path.join(self.key_dir, "public_context.bin"), "wb") as f:
            f.write(public_context_data)

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
        return ts.ckks_vector(self.context, data)

    def decrypt(self, encrypted_data):
        """
        Decrypt CKKS encrypted data.

        Args:
            encrypted_data: Encrypted data
        """
        result = encrypted_data.decrypt()
        return result[0] if len(result) == 1 else result

    @classmethod
    def load_private_context(cls, key_dir="crypto_keys"):
        """
        Load a private context with all keys from a file.

        Args:
            key_dir: Directory where keys are stored

        Returns:
            An instance of HomomorphicEncryptionService with loaded context
        """
        instance = cls.__new__(cls)
        instance.key_dir = key_dir

        with open(os.path.join(key_dir, "private_context.bin"), "rb") as f:
            context_data = f.read()

        instance.context = ts.Context.load(context_data)
        return instance

    @classmethod
    def load_public_context(cls, key_dir="crypto_keys"):
        """
        Load a public context (without secret key) from a file.

        Args:
            key_dir: Directory where keys are stored

        Returns:
            An instance of HomomorphicEncryptionService with loaded context
        """
        instance = cls.__new__(cls)
        instance.key_dir = key_dir

        with open(os.path.join(key_dir, "public_context.bin"), "rb") as f:
            context_data = f.read()

        instance.context = ts.Context.load(context_data)
        return instance


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
    """
    # Initialize the encryption service
    he_service = HomomorphicEncryptionService()

    # Reshape data for LSTM
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Encrypt training data (batches)
    print("Encrypting training data samples...")
    encrypted_X_train = []
    for i in range(min(100, len(X_train_reshaped))):  # Encrypt first 100 samples
        encrypted_sample = []
        for feature in X_train_reshaped[i][0]:
            encrypted_sample.append(he_service.encrypt(feature))
        encrypted_X_train.append(encrypted_sample)

    print(f"Encrypted {len(encrypted_X_train)} training samples")

    # Instead of saving the entire service, just save the path to the keys
    # Create a minimal object that can be pickled
    service_info = {
        "key_dir": he_service.key_dir,
        "scheme": ts.SCHEME_TYPE.CKKS,
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "encryption_type": ts.ENCRYPTION_TYPE.SYMMETRIC,
    }
    joblib.dump(service_info, "model_ckks/encryption_service_info.pkl")

    # Rest of your training code...
    # Build model
    model = build_encrypted_compatible_model((X_train.shape[1],))

    # Train model (using unencrypted data for training as homomorphic operations are limited)
    print("Training model with regular data...")
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

    # Test model on encrypted data
    print("Testing model on encrypted data...")
    # Encrypt a small subset of test data for demonstration
    encrypted_test_samples = min(10, len(X_test_reshaped))
    encrypted_results = []

    for i in range(encrypted_test_samples):
        # Encrypt sample
        encrypted_sample = []
        for feature in X_test_reshaped[i][0]:
            encrypted_sample.append(he_service.encrypt(feature))

        # For demonstration, we'll decrypt before prediction
        # In a real scenario, we'd perform homomorphic operations
        decrypted_sample = np.array([[he_service.decrypt(enc_feat) for enc_feat in encrypted_sample]])
        decrypted_sample = np.reshape(decrypted_sample, (1, 1, len(decrypted_sample[0])))

        # Make prediction
        pred = model.predict(decrypted_sample)[0][0]
        encrypted_results.append((pred, y_test.iloc[i]))

    print(f"Encrypted test results (first {encrypted_test_samples} samples):")
    for i, (pred, actual) in enumerate(encrypted_results):
        print(f"Sample {i+1}: Predicted {pred:.4f}, Actual {actual}")

    # Standard evaluation
    print("Evaluating model on standard test data...")
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


def encrypt_predict(model, X, he_service=None):
    """
    Encrypt data, make prediction, and decrypt result.
    """
    # Load encryption service if not provided
    if he_service is None:
        service_info = joblib.load("model_ckks/encryption_service_info.pkl")
        he_service = HomomorphicEncryptionService(
            scheme=service_info["scheme"],
            poly_modulus_degree=service_info["poly_modulus_degree"],
            coeff_mod_bit_sizes=service_info["coeff_mod_bit_sizes"],
            encryption_type=service_info["encryption_type"],
            key_dir=service_info["key_dir"],
        )

    # Reshape input
    if len(X.shape) == 1:
        X = np.reshape(X, (1, 1, X.shape[0]))
    elif len(X.shape) == 2:
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # Encrypt input
    encrypted_X = []
    for i in range(X.shape[0]):
        encrypted_sample = []
        for feature in X[i][0]:
            encrypted_sample.append(he_service.encrypt(feature))
        encrypted_X.append(encrypted_sample)

    # For demonstration, decrypt before prediction
    # In a real-world scenario, we would perform homomorphic operations
    results = []
    for sample in encrypted_X:
        decrypted_sample = np.array([[he_service.decrypt(enc_feat) for enc_feat in sample]])
        decrypted_sample = np.reshape(decrypted_sample, (1, 1, len(decrypted_sample[0])))
        pred = model.predict(decrypted_sample)[0][0]
        results.append(pred)

    return np.array(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the fraud detection dataset")
    parser.add_argument("--role", type=str, required=True, help="AWS IAM role for SageMaker")

    args = parser.parse_args()

    # Load and process data
    X_train, X_test, y_train, y_test, features = load_data(args.data_path)

    # Train the model
    model = train_encrypted_model(X_train, y_train, X_test, y_test, features)
