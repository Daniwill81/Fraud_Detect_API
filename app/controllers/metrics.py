from fastapi import HTTPException

from app.models import Metrics
from app.serializers.core.metrics import MetricsSerializer


async def get_all_metrics() -> list[MetricsSerializer]:
    try:
        metrics = await Metrics.find_all().to_list()
        return [
            MetricsSerializer(
                accuracy=m.accuracy,
                precision=m.precision,
                recall=m.recall,
                f1_score=m.f1_score,
                auc=m.auc,
                false_positive_rate=m.false_positive_rate,
                false_negative_rate=m.false_negative_rate,
                ckks_or_standard=m.ckks_or_standard,
                timestamp=m.timestamp,
            )
            for m in metrics
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")


async def get_metrics_by_ckks_or_standard(ckks_or_standard: str) -> list[MetricsSerializer]:
    try:
        metrics = await Metrics.find(Metrics.ckks_or_standard == ckks_or_standard).to_list()
        return [
            MetricsSerializer(
                accuracy=m.accuracy,
                precision=m.precision,
                recall=m.recall,
                f1_score=m.f1_score,
                auc=m.auc,
                false_positive_rate=m.false_positive_rate,
                false_negative_rate=m.false_negative_rate,
                ckks_or_standard=m.ckks_or_standard,
                timestamp=m.timestamp,
            )
            for m in metrics
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving metrics for model type {ckks_or_standard}: {str(e)}"
        )


async def get_latest_metrics_comparison() -> dict:
    try:
        standard_metrics = (
            await Metrics.find(Metrics.ckks_or_standard == "standard").sort("-timestamp").limit(1).to_list()
        )
        ckks_metrics = await Metrics.find(Metrics.ckks_or_standard == "ckks").sort("-timestamp").limit(1).to_list()

        result = {
            "standard": standard_metrics[0].model_dump() if standard_metrics else None,
            "ckks": ckks_metrics[0].model_dump() if ckks_metrics else None,
        }

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing metrics: {str(e)}")
