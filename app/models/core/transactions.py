from datetime import datetime

from beanie import Document
from pydantic import Field


class Transactions(Document):
    data: list[float]
    encrypted: bool
    prediction: float
    confidence: float
    institution: str
    is_fraud: bool
    model_type: str  # "standard" or "ckks"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Settings:
        name = "transactions"

    @classmethod
    async def count_by_fraud_status(cls):
        """Return the count of fraudulent and non-fraudulent transactions"""
        pipeline = [
            {"$group": {"_id": "$is_fraud", "count": {"$sum": 1}}},
        ]
        result = await cls.aggregate(pipeline).to_list()
        counts = {str(item["_id"]): item["count"] for item in result}
        return {"fraudulent": counts.get("True", 0), "non_fraudulent": counts.get("False", 0)}
