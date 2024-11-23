"""
AppMain.

This package contains configuration for the project.
The `router` list routes URLS to any accessible endpoint for a app.
The router in routes.py is referred as the main router for this app.
"""


from fastapi import APIRouter

from .train import router as router_train

router_api = APIRouter(redirect_slashes=True)

router_api.include_router(router_train, prefix="/train", tags=["train"])
