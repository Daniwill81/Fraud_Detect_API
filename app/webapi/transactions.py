from fastapi import APIRouter, status

from app.controllers.transactions import get_transaction_counts, test_with_encryption, test_without_encryption
from app.serializers.core.transactions import (
    TransactionCountSerializer,
    TransactionSerializer,
    WriteTransactionSerializer,
)

router = APIRouter()


@router.post("/test_standard/", status_code=status.HTTP_201_CREATED)
async def test_standard(serializer_write: WriteTransactionSerializer) -> TransactionSerializer:
    """
    Test a transaction using the standard model without encryption.
    Results are stored and metrics are updated.
    """
    return await test_without_encryption(serializer_write)


@router.post("/test_ckks/", status_code=status.HTTP_201_CREATED)
async def test_ckks(serializer_write: WriteTransactionSerializer) -> TransactionSerializer:
    """
    Test a transaction using the CKKS model with encryption.
    Results are stored, metrics are updated, and hyperparameters are adjusted.
    """
    return await test_with_encryption(serializer_write)


@router.get("/counts/", status_code=status.HTTP_200_OK)
async def get_counts() -> TransactionCountSerializer:
    """
    Get counts of fraudulent and non-fraudulent transactions.
    """
    return await get_transaction_counts()
