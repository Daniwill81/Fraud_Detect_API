from beanie import Document
from datetime import datetime
from pydantic import Field


class Metrics(Document):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    false_positive_rate: float
    false_negative_rate: float
    model_type: str  # "standard" or "ckks"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Settings:
        name = "metrics"
