import argparse
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Adjust path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import train_ckks_model as ckks_model
import train_standard_model as standard_model


def compare_model_performance(standard_model_obj, encrypted_model_obj, X_test, y_test):
    """
    Compare performances of standard and encrypted models.
    """
    # Reshape for LSTM
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Predictions
    standard_pred = (standard_model_obj.predict(X_test_reshaped) > 0.5).astype(int)
    encrypted_pred = (encrypted_model_obj.predict(X_test_reshaped) > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"],
        "Standard Model": [
            accuracy_score(y_test, standard_pred),
            precision_score(y_test, standard_pred),
            recall_score(y_test, standard_pred),
            f1_score(y_test, standard_pred),
            roc_auc_score(y_test, standard_model_obj.predict(X_test_reshaped)),
        ],
        "Encrypted Model": [
            accuracy_score(y_test, encrypted_pred),
            precision_score(y_test, encrypted_pred),
            recall_score(y_test, encrypted_pred),
            f1_score(y_test, encrypted_pred),
            roc_auc_score(y_test, encrypted_model_obj.predict(X_test_reshaped)),
        ],
    }

    return pd.DataFrame(metrics)


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


def analyze_computational_overhead(standard_model_obj, encrypted_model_obj, X_test):
    """
    Analyze computational overhead of models.
    """
    import sys
    import time

    import psutil

    # Reshape test data
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Prediction time measurement
    def measure_prediction_time(model, X_test_reshaped):
        start_time = time.time()
        model.predict(X_test_reshaped)
        return (time.time() - start_time) * 1000  # in milliseconds

    # Memory usage measurement
    def measure_memory_usage(model):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # in MB

    overhead_metrics = {
        "Metric": ["Model Size (KB)", "Prediction Time (ms)", "Memory Usage (MB)"],
        "Standard Model": [
            standard_model_obj.count_params() * 4 / 1024,  # Approximate size
            measure_prediction_time(standard_model_obj, X_test_reshaped),
            measure_memory_usage(standard_model_obj),
        ],
        "Encrypted Model": [
            encrypted_model_obj.count_params() * 4 / 1024,  # Approximate size
            measure_prediction_time(encrypted_model_obj, X_test_reshaped),
            measure_memory_usage(encrypted_model_obj),
        ],
    }

    return pd.DataFrame(overhead_metrics)


def visualize_computational_overhead(overhead_df, output_dir="results"):
    """
    Visualize computational overhead.
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    melted_df = pd.melt(overhead_df, id_vars=["Metric"], var_name="Model", value_name="Value")

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Overhead bar plot
    ax = sns.barplot(x="Metric", y="Value", hue="Model", data=melted_df, palette="Set1")

    plt.title("Computational Overhead: Standard vs Encrypted Model", fontsize=15)
    plt.xlabel("Overhead Metrics", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Model Type")

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "computational_overhead.png"))
    plt.close()

    # Save overhead data
    overhead_df.to_csv(os.path.join(output_dir, "computational_overhead.csv"), index=False)


def generate_comparison_report(performance_df, overhead_df, output_dir="results"):
    """
    Generate markdown comparison report.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "model_comparison_report.md"), "w") as f:
        f.write("# Fraud Detection Model Comparison Report\n\n")

        f.write("## Performance Comparison\n\n")
        f.write(performance_df.to_markdown(index=False))

        f.write("\n\n## Computational Overhead\n\n")
        f.write(overhead_df.to_markdown(index=False))

        f.write("\n\n## Analysis of Trade-offs\n\n")
        f.write("### Impact of Encryption\n")
        f.write("- **Performance Loss**: Evaluate the decrease in metrics\n")
        f.write("- **Computational Overhead**: Quantify increase in time and memory\n")
        f.write("- **Confidentiality Gain**: Protection of sensitive data\n")


def main(data_path):
    """
    Main function for model comparison.
    """
    # Load data (using identical preprocessing)
    X_train_standard, X_test_standard, y_train_standard, y_test_standard, features_standard = standard_model.load_data(
        data_path
    )
    X_train_ckks, X_test_ckks, y_train_ckks, y_test_ckks, features_ckks = ckks_model.load_data(data_path)

    # Train models
    standard_model_obj = standard_model.train_model(
        X_train_standard, y_train_standard, X_test_standard, y_test_standard, features_standard
    )
    encrypted_model_obj = ckks_model.train_encrypted_model(
        X_train_ckks, y_train_ckks, X_test_ckks, y_test_ckks, features_ckks
    )

    # Performance comparison
    performance_df = compare_model_performance(
        standard_model_obj, encrypted_model_obj, X_test_standard, y_test_standard
    )
    visualize_performance_comparison(performance_df)

    # Computational overhead analysis
    overhead_df = analyze_computational_overhead(standard_model_obj, encrypted_model_obj, X_test_standard)
    visualize_computational_overhead(overhead_df)

    # Generate report
    generate_comparison_report(performance_df, overhead_df)

    print("Comparison completed. Check the 'results' folder for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection Model Comparison")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--role", type=str, required=True, help="AWS IAM role for SageMaker")
    args = parser.parse_args()

    main(args.data_path)
