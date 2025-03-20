from pydantic import BaseModel

class TransactionRequest(BaseModel):
    data: List[float]
    encrypt: bool = False  # Whether to use encryption
    institution: str  # Name of the financial institution

class TransactionResponse(BaseModel):
    prediction: float
    confidence: float
    is_fraud: bool

class TransactionCountResponse(BaseModel):
    fraudulent: int
    non_fraudulent: int
