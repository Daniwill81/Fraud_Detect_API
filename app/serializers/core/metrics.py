from pydantic import BaseModel


class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    false_positive_rate: float
    false_negative_rate: float
    model_type: str
    timestamp: str
