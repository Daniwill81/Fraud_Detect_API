from datetime import datetime
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from app.models import Hyperparameters, Metrics, Transactions
from AppMain.settings import AppSettings


async def update_metrics(transactions: Transactions, ckks_or_standard: str) -> None:
    """
    Update metrics after a transaction is processed.

    Args:
        transaction: The transaction object
        ckks_or_standard: Type of model used ("standard" or "ckks")
    """
    # Fetch all transactions for this model type
    transactions = await Transactions.find(Transactions.ckks_or_standard == ckks_or_standard).to_list()

    # Skip if not enough transactions
    if len(transactions) < 5:
        return

    # Extract true labels and predictions
    y_true = [t.is_fraud for t in transactions]
    y_pred = [t.prediction > AppSettings.FRAUD_THRESHOLD for t in transactions]
    y_scores = [t.prediction for t in transactions]

    # Calculate metrics
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Handle case with only one class
        if len(set(y_true)) < 2:
            auc = 0.5  # Default value when only one class
        else:
            auc = roc_auc_score(y_true, y_scores)

        # Calculate false positive/negative rates
        false_positive = sum((np.array(y_pred) == 1) & (np.array(y_true) == 0))
        false_negative = sum((np.array(y_pred) == 0) & (np.array(y_true) == 1))

        true_negatives = sum(np.array(y_true) == 0)
        true_positives = sum(np.array(y_true) == 1)

        false_positive_rate = false_positive / max(1, true_negatives)
        false_negative_rate = false_negative / max(1, true_positives)

        # Save metrics to MongoDB
        metrics = Metrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            auc=float(auc),
            false_positive_rate=float(false_positive_rate),
            false_negative_rate=float(false_negative_rate),
            ckks_or_standard=ckks_or_standard,
        )
        await metrics.insert()
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")


async def update_hyperparameters(result: dict[str, Any]) -> None:
    """
    Update hyperparameters based on model performance.
    Only for the CKKS model.

    Args:
        result: Model prediction result
    """
    # Fetch current hyperparameters
    hyperparams = await Hyperparameters.find_one(Hyperparameters.ckks_or_standard == "ckks")
    if not hyperparams:
        # Initialize with default values if not exists
        hyperparams = Hyperparameters(learning_rate=0.001, batch_size=32, epochs=10, ckks_or_standard="ckks")

    # Update hyperparameters based on model confidence
    if result["confidence"] < 0.7:
        # Lower confidence means we need to adjust hyperparameters
        hyperparams.learning_rate *= 0.95  # Reduce learning rate
        hyperparams.batch_size = min(hyperparams.batch_size * 2, 128)  # Increase batch size
        hyperparams.epochs += 1  # Add an epoch
    elif result["confidence"] > 0.95:
        # Very high confidence might mean overfitting
        hyperparams.learning_rate *= 1.05  # Increase learning rate
        hyperparams.batch_size = max(hyperparams.batch_size // 2, 16)  # Decrease batch size

    # Save updated hyperparameters
    hyperparams.last_updated = datetime.now().isoformat()
    await hyperparams.save()
