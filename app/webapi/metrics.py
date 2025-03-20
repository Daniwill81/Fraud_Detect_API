from fastapi import APIRouter
from typing import List, Dict
from app.controllers.metrics_controller import get_all_metrics, get_metrics_by_model_type, get_latest_metrics_comparison
from app.schemas.transaction_schemas import MetricsResponse

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/", response_model=List[MetricsResponse])
async def endpoint_get_all_metrics():
    """
    Get all metrics for both models.
    """
    return await get_all_metrics()

@router.get("/standard", response_model=List[MetricsResponse])
async def endpoint_get_standard_metrics():
    """
    Get metrics only for the standard model.
    """
    return await get_metrics_by_model_type("standard")

@router.get("/ckks", response_model=List[MetricsResponse])
async def endpoint_get_ckks_metrics():
    """
    Get metrics only for the CKKS model.
    """
    return await get_metrics_by_model_type("ckks")

@router.get("/compare", response_model=Dict)
async def endpoint_compare_metrics():
    """
    Compare the latest metrics between standard and CKKS models.
    """
    return await get_latest_metrics_comparison()
