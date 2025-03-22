import datetime

from sap.fastapi.serializers import ObjectSerializer, WriteObjectSerializer

from app.models import Hyperparameters


class HyperparametersSerializer(ObjectSerializer[Hyperparameters]):
    learning_rate: float
    batch_size: int
    epochs: int
    model_type: str
    last_updated: datetime.datetime
