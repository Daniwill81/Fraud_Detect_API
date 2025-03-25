from datetime import datetime

from beanie import Document
from pydantic import Field


class Metrics(Document):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    false_positive_rate: float
    false_negative_rate: float
    ckks_or_standard: str  # "standard" or "ckks"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Settings:
        name = "metrics"
