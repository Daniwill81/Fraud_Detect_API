from fastapi import HTTPException

from app.models import Transactions
from app.serializers.core.transactions import TransactionSerializer, WriteTransactionSerializer
from app.services.metrics import update_hyperparameters, update_metrics
from app.services.sagemaker import predict_ckks, predict_standard
from AppMain.settings import AppSettings
from utils.encryption import encrypt_data


async def test_without_encryption(transaction: WriteTransactionSerializer) -> TransactionSerializer:
    try:
        # Send data to SageMaker standard model
        result = await predict_standard(transaction.data)

        # Determine if transaction is fraudulent
        is_fraud = result["prediction"] > AppSettings.FRAUD_THRESHOLD

        # Save result to MongoDB
        transaction_db = Transactions(
            data=transaction.data,
            encrypted=False,
            prediction=result["prediction"],
            confidence=result["confidence"],
            institution=transaction.institution,
            is_fraud=is_fraud,
            model_type="standard",
        )
        await transaction_db.insert()

        # Update metrics for standard model
        await update_metrics(transaction_db, "standard")

        return TransactionSerializer(
            prediction=result["prediction"], confidence=result["confidence"], is_fraud=is_fraud
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing transaction: {str(e)}")


async def test_with_encryption(transaction: WriteTransactionSerializer) -> TransactionSerializer:
    try:
        # Encrypt data
        encrypted_data = encrypt_data(transaction.data)

        # Send encrypted data to SageMaker CKKS model
        result = await predict_ckks(encrypted_data)

        # Determine if transaction is fraudulent
        is_fraud = result["prediction"] > AppSettings.FRAUD_THRESHOLD

        # Save result to MongoDB
        transaction_db = Transactions(
            data=transaction.data,
            encrypted=True,
            prediction=result["prediction"],
            confidence=result["confidence"],
            institution=transaction.institution,
            is_fraud=is_fraud,
            model_type="ckks",
        )
        await transaction_db.insert()

        # Update metrics for CKKS model
        await update_metrics(transaction_db, "ckks")

        # Update hyperparameters only for CKKS model
        await update_hyperparameters(result)

        return TransactionSerializer(
            prediction=result["prediction"], confidence=result["confidence"], is_fraud=is_fraud
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing encrypted transaction: {str(e)}")


async def run_demo(transaction: WriteTransactionSerializer) -> TransactionSerializer:
    try:
        if transaction.encrypt:
            return await test_with_encryption(transaction)
        return await test_without_encryption(transaction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in demo: {str(e)}")


async def get_transaction_counts() -> dict:
    try:
        counts = await Transactions.count_by_fraud_status()
        return counts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting transaction counts: {str(e)}")
