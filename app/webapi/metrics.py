from fastapi import APIRouter, status

from app.controllers.metrics import get_all_metrics, get_latest_metrics_comparison, get_metrics_by_model_type
from app.serializers.core.metrics import MetricsSerializer

router = APIRouter()


@router.get("/", status_code=status.HTTP_200_OK)
async def all_metrics() -> list[MetricsSerializer]:
    """
    Get all metrics for both models.
    """
    return await get_all_metrics()


@router.get("/standard/", status_code=status.HTTP_200_OK)
async def standard_metrics() -> list[MetricsSerializer]:
    """
    Get metrics only for the standard model.
    """
    return await get_metrics_by_model_type("standard")


@router.get("/ckks/", status_code=status.HTTP_200_OK)
async def ckks_metrics() -> list[MetricsSerializer]:
    """
    Get metrics only for the CKKS model.
    """
    return await get_metrics_by_model_type("ckks")


@router.get("/compare/", status_code=status.HTTP_200_OK)
async def compare_two_models_metrics() -> dict:
    """
    Compare the latest metrics between standard and CKKS models.
    """
    return await get_latest_metrics_comparison()
