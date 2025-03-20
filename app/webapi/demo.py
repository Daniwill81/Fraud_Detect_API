from fastapi import APIRouter, Depends
from app.schemas.transaction_schemas import TransactionRequest, TransactionResponse, TransactionCountResponse
from app.controllers.transaction_controller import (
    test_without_encryption,
    test_with_encryption,
    run_demo,
    get_transaction_counts
)
from typing import Dict

@router.post("/demo", response_model=TransactionResponse)
async def endpoint_demo(transaction: TransactionRequest):
    """
    Demo endpoint that can use either model based on the 'encrypt' flag.
    """
    return await run_demo(transaction)
