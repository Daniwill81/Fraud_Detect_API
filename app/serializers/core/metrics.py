from sap.fastapi.serializers import ObjectSerializer

from app.models import Metrics


class MetricsSerializer(ObjectSerializer[Metrics]):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    false_positive_rate: float
    false_negative_rate: float
    model_type: str
    timestamp: str
