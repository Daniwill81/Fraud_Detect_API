import argparse
import gc
import os
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import tenseal as ts
from keras import models
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Adjust path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import train_ckks_model as ckks_model
import train_standard_model as standard_model
from train_ckks_model import HomomorphicEncryptionService


def compare_model_performance(standard_model_obj, encrypted_model_obj, X_test, y_test, he_service):
    """
    Compare performances of standard and encrypted models.
    """
    # Charger les informations du service de chiffrement
    service_info = joblib.load("model_ckks/encryption_service_info.pkl")

    # Recréer le service de chiffrement
    he_service = HomomorphicEncryptionService(
        scheme=service_info["scheme"],
        poly_modulus_degree=service_info["poly_modulus_degree"],
        coeff_mod_bit_sizes=service_info["coeff_mod_bit_sizes"],
        encryption_type=service_info["encryption_type"],
        key_dir=service_info["key_dir"],
    )

    # Reshape for LSTM
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Get predictions
    print("Making predictions with standard model...")
    standard_pred_proba = standard_model_obj.predict(X_test_reshaped)

    print("Making predictions with encrypted model...")
    encrypted_pred_proba = []

    # For demonstration, use a subset of test data
    test_samples = min(100, len(X_test_reshaped))
    for i in range(test_samples):
        # Encrypt sample
        encrypted_sample = []
        for feature in X_test_reshaped[i][0]:
            encrypted_sample.append(he_service.encrypt(feature))

        # Decrypt for prediction
        decrypted_sample = np.array([[he_service.decrypt(enc_feat) for enc_feat in encrypted_sample]])
        decrypted_sample = np.reshape(decrypted_sample, (1, 1, len(decrypted_sample[0])))

        # Make prediction
        pred = encrypted_model_obj.predict(decrypted_sample)[0][0]
        encrypted_pred_proba.append(pred)

    # Convert to numpy array and use only the subset for comparison
    encrypted_pred_proba = np.array(encrypted_pred_proba).reshape(-1, 1)
    y_test_subset = y_test.iloc[:test_samples]
    standard_pred_proba_subset = standard_pred_proba[:test_samples]

    # Convert to binary predictions
    standard_pred = (standard_pred_proba_subset > 0.5).astype(int)
    encrypted_pred = (encrypted_pred_proba > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"],
        "Standard Model": [
            accuracy_score(y_test_subset, standard_pred),
            precision_score(y_test_subset, standard_pred),
            recall_score(y_test_subset, standard_pred),
            f1_score(y_test_subset, standard_pred),
            roc_auc_score(y_test_subset, standard_pred_proba_subset),
        ],
        "Encrypted Model": [
            accuracy_score(y_test_subset, encrypted_pred),
            precision_score(y_test_subset, encrypted_pred),
            recall_score(y_test_subset, encrypted_pred),
            f1_score(y_test_subset, encrypted_pred),
            roc_auc_score(y_test_subset, encrypted_pred_proba),
        ],
    }

    prediction_data = {
        "standard_pred": standard_pred,
        "standard_pred_proba": standard_pred_proba_subset,
        "encrypted_pred": encrypted_pred,
        "encrypted_pred_proba": encrypted_pred_proba,
        "y_test_subset": y_test_subset,
    }

    return pd.DataFrame(metrics), prediction_data


def visualize_performance_comparison(comparison_df, output_dir="results"):
    """
    Create comparative performance visualizations.
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    melted_df = pd.melt(comparison_df, id_vars=["Metric"], var_name="Model", value_name="Value")

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Comparative bar plot
    ax = sns.barplot(x="Metric", y="Value", hue="Model", data=melted_df, palette="Set2")

    plt.title("Performance Comparison: Standard vs Encrypted Model", fontsize=15)
    plt.xlabel("Performance Metrics", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Model Type")

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_comparison.png"))
    plt.close()

    # Save comparison data
    comparison_df.to_csv(os.path.join(output_dir, "performance_metrics.csv"), index=False)


def visualize_confusion_matrices(prediction_data, output_dir="results"):
    """
    Visualize confusion matrices for both models.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get confusion matrices
    standard_cm = confusion_matrix(prediction_data["y_test_subset"], prediction_data["standard_pred"])
    encrypted_cm = confusion_matrix(prediction_data["y_test_subset"], prediction_data["encrypted_pred"])

    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot standard model confusion matrix
    sns.heatmap(
        standard_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax1,
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
    )
    ax1.set_title("Standard Model Confusion Matrix", fontsize=14)
    ax1.set_xlabel("Predicted Label", fontsize=12)
    ax1.set_ylabel("True Label", fontsize=12)

    # Plot encrypted model confusion matrix
    sns.heatmap(
        encrypted_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax2,
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
    )
    ax2.set_title("Encrypted Model Confusion Matrix", fontsize=14)
    ax2.set_xlabel("Predicted Label", fontsize=12)
    ax2.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"))
    plt.close()


def visualize_roc_curves(prediction_data, output_dir="results"):
    """
    Visualize ROC curves for both models.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calculate ROC curve points
    standard_fpr, standard_tpr, _ = roc_curve(prediction_data["y_test_subset"], prediction_data["standard_pred_proba"])
    encrypted_fpr, encrypted_tpr, _ = roc_curve(
        prediction_data["y_test_subset"], prediction_data["encrypted_pred_proba"]
    )

    # Calculate AUC scores
    standard_auc = roc_auc_score(prediction_data["y_test_subset"], prediction_data["standard_pred_proba"])
    encrypted_auc = roc_auc_score(prediction_data["y_test_subset"], prediction_data["encrypted_pred_proba"])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(standard_fpr, standard_tpr, label=f"Standard Model (AUC = {standard_auc:.3f})", color="blue", linewidth=2)
    plt.plot(
        encrypted_fpr, encrypted_tpr, label=f"Encrypted Model (AUC = {encrypted_auc:.3f})", color="red", linewidth=2
    )

    # Add diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")

    # Add labels and title
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves: Standard vs Encrypted Model", fontsize=14)

    # Add legend and grid
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Set limits
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()


def analyze_computational_overhead(standard_model_obj, encrypted_model_obj, X_test, he_service):
    """
    Analyze computational overhead of models.
    """

    # Reshape test data
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Define sample sizes for testing
    sample_sizes = [1, 5, 10, 25, 50]
    results = {
        "sample_size": [],
        "standard_time": [],
        "standard_memory": [],
        "encrypted_time": [],
        "encrypted_memory": [],
        "encryption_time": [],
        "decryption_time": [],
    }

    for n_samples in sample_sizes:
        test_subset = X_test_reshaped[:n_samples]
        print(f"Testing with {n_samples} samples...")

        # === Standard model ===
        # Memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # in MB

        # Time prediction
        start_time = time.time()
        standard_model_obj.predict(test_subset)
        standard_time = time.time() - start_time

        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024
        standard_memory = mem_after - mem_before

        # === Encrypted model ===
        # Reset memory measurement
        _ = gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024

        # Measure encryption time
        start_encrypt_time = time.time()
        encrypted_samples = []
        for i in range(n_samples):
            encrypted_sample = []
            for feature in test_subset[i][0]:
                encrypted_sample.append(he_service.encrypt(feature))
            encrypted_samples.append(encrypted_sample)
        encryption_time = time.time() - start_encrypt_time

        # Measure decryption and prediction time
        start_pred_time = time.time()
        for sample in encrypted_samples:
            # Decrypt
            decrypted_sample = np.array([[he_service.decrypt(enc_feat) for enc_feat in sample]])
            decrypted_sample = np.reshape(decrypted_sample, (1, 1, len(decrypted_sample[0])))

            # Predict
            _ = encrypted_model_obj.predict(decrypted_sample)

        encrypted_time = time.time() - start_pred_time

        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024
        encrypted_memory = mem_after - mem_before

        # Measure decryption time separately
        start_decrypt_time = time.time()
        for sample in encrypted_samples:
            _ = [he_service.decrypt(enc_feat) for enc_feat in sample]
        decryption_time = time.time() - start_decrypt_time

        # Record results
        results["sample_size"].append(n_samples)
        results["standard_time"].append(standard_time)
        results["standard_memory"].append(standard_memory)
        results["encrypted_time"].append(encrypted_time)
        results["encrypted_memory"].append(encrypted_memory)
        results["encryption_time"].append(encryption_time)
        results["decryption_time"].append(decryption_time)

    return pd.DataFrame(results)


def visualize_computational_overhead(overhead_df, output_dir="results"):
    """
    Visualize computational overhead comparison.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Time comparison
    plt.figure(figsize=(12, 6))
    plt.plot(overhead_df["sample_size"], overhead_df["standard_time"], marker="o", label="Standard Model")
    plt.plot(overhead_df["sample_size"], overhead_df["encrypted_time"], marker="s", label="Encrypted Model (Total)")
    plt.plot(overhead_df["sample_size"], overhead_df["encryption_time"], marker="^", label="Encryption Time")
    plt.plot(overhead_df["sample_size"], overhead_df["decryption_time"], marker="*", label="Decryption Time")

    plt.title("Computational Time Comparison", fontsize=14)
    plt.xlabel("Number of Samples", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_comparison.png"))
    plt.close()

    # Memory comparison
    plt.figure(figsize=(12, 6))
    plt.plot(overhead_df["sample_size"], overhead_df["standard_memory"], marker="o", label="Standard Model")
    plt.plot(overhead_df["sample_size"], overhead_df["encrypted_memory"], marker="s", label="Encrypted Model")

    plt.title("Memory Usage Comparison", fontsize=14)
    plt.xlabel("Number of Samples", fontsize=12)
    plt.ylabel("Memory Usage (MB)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_comparison.png"))
    plt.close()

    # Save results
    overhead_df.to_csv(os.path.join(output_dir, "computational_overhead.csv"), index=False)


def compare_model_architecture(standard_model_obj, encrypted_model_obj, output_dir="results"):
    """
    Compare model architectures and parameters between standard and encrypted models.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get model summaries
    standard_summary = []
    encrypted_summary = []

    # Redirect stdout to capture summaries
    import sys
    from io import StringIO

    # Capture standard model summary
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    standard_model_obj.summary()
    sys.stdout = old_stdout
    standard_summary = mystdout.getvalue()

    # Capture encrypted model summary
    sys.stdout = mystdout = StringIO()
    encrypted_model_obj.summary()
    sys.stdout = old_stdout
    encrypted_summary = mystdout.getvalue()

    # Save summaries
    with open(os.path.join(output_dir, "model_architecture_comparison.txt"), "w") as f:
        f.write("Standard Model Architecture:\n")
        f.write("-" * 50 + "\n")
        f.write(standard_summary)
        f.write("\n\n")
        f.write("Encrypted Model Architecture:\n")
        f.write("-" * 50 + "\n")
        f.write(encrypted_summary)

    # Count trainable parameters
    standard_params = standard_model_obj.count_params()
    encrypted_params = encrypted_model_obj.count_params()

    # Create comparison DataFrame
    comparison = pd.DataFrame(
        {
            "Model Type": ["Standard Model", "Encrypted Model"],
            "Trainable Parameters": [standard_params, encrypted_params],
        }
    )

    # Save comparison
    comparison.to_csv(os.path.join(output_dir, "model_parameters.csv"), index=False)

    return comparison


def main():
    """
    Main function to run model comparisons.
    """
    parser = argparse.ArgumentParser(description="Compare standard and encrypted fraud detection models")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the fraud detection dataset")
    parser.add_argument("--role", type=str, required=True, help="AWS IAM role for SageMaker")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    # Load data using the standard model loader (both use same preprocessing)
    X_train, X_test, y_train, y_test, features = standard_model.load_data(args.data_path)

    # Load models
    print("Loading standard model...")
    standard_model_obj = models.load_model("model_standard/standard_model.keras")

    print("Loading encrypted model...")
    encrypted_model_obj = models.load_model("model_ckks/ckks_model.keras")

    # Load encryption service INFO (not the service itself)
    service_info = joblib.load("model_ckks/encryption_service_info.pkl")

    # Recreate the encryption service
    he_service = HomomorphicEncryptionService(
        scheme=service_info["scheme"],
        poly_modulus_degree=service_info["poly_modulus_degree"],
        coeff_mod_bit_sizes=service_info["coeff_mod_bit_sizes"],
        encryption_type=service_info["encryption_type"],
        key_dir=service_info["key_dir"],
    )

    # Now pass this properly initialized service to your functions
    comparison_df, prediction_data = compare_model_performance(
        standard_model_obj, encrypted_model_obj, X_test, y_test, he_service
    )

    # Visualize performance comparison
    print("Visualizing performance metrics...")
    visualize_performance_comparison(comparison_df, args.output_dir)
    visualize_confusion_matrices(prediction_data, args.output_dir)
    visualize_roc_curves(prediction_data, args.output_dir)

    # Analyze computational overhead
    print("Analyzing computational overhead...")
    overhead_df = analyze_computational_overhead(standard_model_obj, encrypted_model_obj, X_test, he_service)
    visualize_computational_overhead(overhead_df, args.output_dir)

    # Compare model architectures
    print("Comparing model architectures...")
    compare_model_architecture(standard_model_obj, encrypted_model_obj, args.output_dir)

    print(f"All comparisons completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
