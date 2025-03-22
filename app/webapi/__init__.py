"""
AppMain.

This package contains configuration for the project.
The `router` list routes URLS to any accessible endpoint for a app.
The router in routes.py is referred as the main router for this app.
"""

from fastapi import APIRouter

from .auth import router as router_auth
from .data_cleaning import router as router_data_cleaning
from .demo import router as router_demo
from .metrics import router as router_metrics
from .transactions import router as router_transactions
from .user import router as router_user

router_api = APIRouter(redirect_slashes=True)

router_api.include_router(router_demo, prefix="/demo", tags=["demo"])
router_api.include_router(router_transactions, prefix="/transactions", tags=["transactions"])
router_api.include_router(router_auth, prefix="/auth/user_token", tags=["auth"])
router_api.include_router(router_metrics, prefix="/metrics", tags=["metrics"])
router_api.include_router(router_data_cleaning, prefix="/data-cleaning", tags=["data_cleaning"])
router_api.include_router(router_user, prefix="/user", tags=["user"])
