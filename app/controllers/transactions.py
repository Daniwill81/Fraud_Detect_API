from fastapi import HTTPException
from datetime import datetime
from app.models.mongodb_models import Transaction, Hyperparameters
from app.schemas.transaction_schemas import TransactionRequest, TransactionResponse
from app.core.encryption import encrypt_data
from app.core.config import settings
from app.services.sagemaker_service import predict_standard, predict_ckks
from app.services.metrics_service import update_metrics, update_hyperparameters
from typing import List, Dict, Any

async def test_without_encryption(transaction: TransactionRequest) -> TransactionResponse:
    try:
        # Send data to SageMaker standard model
        result = await predict_standard(transaction.data)
        
        # Determine if transaction is fraudulent
        is_fraud = result["prediction"] > settings.FRAUD_THRESHOLD
        
        # Save result to MongoDB
        transaction_db = Transaction(
            data=transaction.data,
            encrypted=False,
            prediction=result["prediction"],
            confidence=result["confidence"],
            institution=transaction.institution,
            is_fraud=is_fraud,
            model_type="standard"
        )
        await transaction_db.insert()
        
        # Update metrics for standard model
        await update_metrics(transaction_db, "standard")
        
        return TransactionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            is_fraud=is_fraud
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing transaction: {str(e)}")

async def test_with_encryption(transaction: TransactionRequest) -> TransactionResponse:
    try:
        # Encrypt data
        encrypted_data = encrypt_data(transaction.data)
        
        # Send encrypted data to SageMaker CKKS model
        result = await predict_ckks(encrypted_data)
        
        # Determine if transaction is fraudulent
        is_fraud = result["prediction"] > settings.FRAUD_THRESHOLD
        
        # Save result to MongoDB
        transaction_db = Transaction(
            data=transaction.data,
            encrypted=True,
            prediction=result["prediction"],
            confidence=result["confidence"],
            institution=transaction.institution,
            is_fraud=is_fraud,
            model_type="ckks"
        )
        await transaction_db.insert()
        
        # Update metrics for CKKS model
        await update_metrics(transaction_db, "ckks")
        
        # Update hyperparameters only for CKKS model
        await update_hyperparameters(result)
        
        return TransactionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            is_fraud=is_fraud
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing encrypted transaction: {str(e)}")

async def run_demo(transaction: TransactionRequest) -> TransactionResponse:
    try:
        if transaction.encrypt:
            return await test_with_encryption(transaction)
        else:
            return await test_without_encryption(transaction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in demo: {str(e)}")

async def get_transaction_counts() -> Dict:
    try:
        counts = await Transaction.count_by_fraud_status()
        return counts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting transaction counts: {str(e)}")
