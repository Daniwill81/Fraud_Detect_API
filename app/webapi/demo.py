from fastapi import APIRouter, status

from app.controllers.transactions import run_demo
from app.serializers.core.transactions import TransactionSerializer, WriteTransactionSerializer

router = APIRouter()


@router.post("/", status_code=status.HTTP_201_CREATED)
async def demo(serializer_write: WriteTransactionSerializer) -> TransactionSerializer:
    """
    Demo endpoint that can use either model based on the 'encrypt' flag.
    """
    return await run_demo(serializer_write)
