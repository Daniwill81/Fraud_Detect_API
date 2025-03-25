from datetime import datetime

from beanie import Document
from pydantic import Field


class Hyperparameters(Document):
    learning_rate: float
    batch_size: int
    epochs: int
    ckks_or_standard: str = "ckks"  # Only for CKKS model
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Settings:
        name = "hyperparameters"
