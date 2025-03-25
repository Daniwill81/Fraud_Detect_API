import datetime

from sap.fastapi.serializers import ObjectSerializer

from app.models import Hyperparameters


class HyperparametersSerializer(ObjectSerializer[Hyperparameters]):
    id: str
    learning_rate: float
    batch_size: int
    epochs: int
    ckks_or_standard: str
    last_updated: datetime.datetime
