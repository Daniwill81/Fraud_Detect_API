from sap.fastapi.serializers import ObjectSerializer, WriteObjectSerializer

from app.models import Transactions


class WriteTransactionSerializer(WriteObjectSerializer[Transactions]):
    data: list[float]
    encrypt: bool = False  # Whether to use encryption
    institution: str  # Name of the financial institution


class TransactionSerializer(ObjectSerializer[Transactions]):
    prediction: float
    confidence: float
    is_fraud: bool


class TransactionCountSerializer(ObjectSerializer[Transactions]):
    fraudulent: int
    non_fraudulent: int
