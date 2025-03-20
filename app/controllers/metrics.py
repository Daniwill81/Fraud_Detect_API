from fastapi import HTTPException
from app.models.mongodb_models import Metrics
from typing import List
from app.schemas.transaction_schemas import MetricsResponse

async def get_all_metrics() -> List[MetricsResponse]:
    try:
        metrics = await Metrics.find_all().to_list()
        return [
            MetricsResponse(
                accuracy=m.accuracy,
                precision=m.precision,
                recall=m.recall,
                f1_score=m.f1_score,
                auc=m.auc,
                false_positive_rate=m.false_positive_rate,
                false_negative_rate=m.false_negative_rate,
                model_type=m.model_type,
                timestamp=m.timestamp
            ) for m in metrics
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")

async def get_metrics_by_model_type(model_type: str) -> List[MetricsResponse]:
    try:
        metrics = await Metrics.find(Metrics.model_type == model_type).to_list()
        return [
            MetricsResponse(
                accuracy=m.accuracy,
                precision=m.precision,
                recall=m.recall,
                f1_score=m.f1_score,
                auc=m.auc,
                false_positive_rate=m.false_positive_rate,
                false_negative_rate=m.false_negative_rate, 
                model_type=m.model_type,
                timestamp=m.timestamp
            ) for m in metrics
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics for model type {model_type}: {str(e)}")

async def get_latest_metrics_comparison() -> dict:
    try:
        standard_metrics = await Metrics.find(Metrics.model_type == "standard").sort("-timestamp").limit(1).to_list()
        ckks_metrics = await Metrics.find(Metrics.model_type == "ckks").sort("-timestamp").limit(1).to_list()
        
        result = {
            "standard": standard_metrics[0].dict() if standard_metrics else None,
            "ckks": ckks_metrics[0].dict() if ckks_metrics else None
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing metrics: {str(e)}")
