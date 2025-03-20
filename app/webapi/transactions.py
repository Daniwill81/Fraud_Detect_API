from fastapi import APIRouter, Depends
from app.schemas.transaction_schemas import TransactionRequest, TransactionResponse, TransactionCountResponse
from app.controllers.transaction_controller import (
    test_without_encryption,
    test_with_encryption,
    run_demo,
    get_transaction_counts
)
from typing import Dict

router = APIRouter(prefix="/transactions", tags=["transactions"])

@router.post("/test-standard", response_model=TransactionResponse)
async def endpoint_test_standard(transaction: TransactionRequest):
    """
    Test a transaction using the standard model without encryption.
    Results are stored and metrics are updated.
    """
    return await test_without_encryption(transaction)

@router.post("/test-ckks", response_model=TransactionResponse)
async def endpoint_test_ckks(transaction: TransactionRequest):
    """
    Test a transaction using the CKKS model with encryption.
    Results are stored, metrics are updated, and hyperparameters are adjusted.
    """
    return await test_with_encryption(transaction)

@router.get("/counts", response_model=TransactionCountResponse)
async def endpoint_get_counts():
    """
    Get counts of fraudulent and non-fraudulent transactions.
    """
    return await get_transaction_counts()
from fastapi import APIRouter, Depends
from app.schemas.transaction_schemas import TransactionRequest, TransactionResponse, TransactionCountResponse
from app.controllers.transaction_controller import (
    test_without_encryption,
    test_with_encryption,
    run_demo,
    get_transaction_counts
)
from typing import Dict

router = APIRouter(prefix="/transactions", tags=["transactions"])

@router.post("/test-standard", response_model=TransactionResponse)
async def endpoint_test_standard(transaction: TransactionRequest):
    """
    Test a transaction using the standard model without encryption.
    Results are stored and metrics are updated.
    """
    return await test_without_encryption(transaction)

@router.post("/test-ckks", response_model=TransactionResponse)
async def endpoint_test_ckks(transaction: TransactionRequest):
    """
    Test a transaction using the CKKS model with encryption.
    Results are stored, metrics are updated, and hyperparameters are adjusted.
    """
    return await test_with_encryption(transaction)

@router.post("/demo", response_model=TransactionResponse)
async def endpoint_demo(transaction: TransactionRequest):
    """
    Demo endpoint that can use either model based on the 'encrypt' flag.
    """
    return await run_demo(transaction)

@router.get("/counts", response_model=TransactionCountResponse)
async def endpoint_get_counts():
    """
    Get counts of fraudulent and non-fraudulent transactions.
    """
    return await get_transaction_counts()
