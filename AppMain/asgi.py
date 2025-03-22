"""
ASGI.

This is MAIN entrypoint to the application.
It exposes the ASGI callable as a module-level variable named ``app``.

"""

import logging
import typing
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from starlette.responses import JSONResponse
from starlette.routing import Mount

from sap.beanie.client import BeanieClient
from sap.fastapi.middleware import InitBeanieMiddleware  # , LogServerErrorMiddleware

from app import models
from app.webapi import router_api

from .settings import AppSettings, logger

# from app.webapi import router_api


@asynccontextmanager
async def lifespan(current_app: FastAPI) -> typing.AsyncGenerator[None, None]:
    """Initialize beanie on startup."""
    assert current_app
    await initialize_beanie()
    # await update_uvicorn_logger()
    yield


# Initialize application
app = FastAPI(docs_url=None, redoc_url=None, title=AppSettings.PROJ_NAME)

# Enable cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount RESTFul API
app_api = FastAPI(
    docs_url=None,
    redoc_url="/doc",
    title=AppSettings.PROJ_NAME,
    description="sfd project API",
)
app_api.include_router(router_api)
app.routes.append(Mount(path="/api/", app=app_api, name="api"))

# Load sub-apps routes and documents
document_models = []

# Retrieve the lists of documents for beanie initialization
for model_name in models.__all__:
    document_models.append(getattr(models, model_name))


# Register middleware
app.add_middleware(
    InitBeanieMiddleware,
    mongo_params=AppSettings.MONGO,
    document_models=document_models,
)


# Events to run on startups
async def initialize_beanie() -> None:
    """Initialize beanie on startup."""
    await BeanieClient.init(mongo_params=AppSettings.MONGO, document_models=document_models)


# Always log exception
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Log all request validation errors to a file."""
    logger.exception(exc.errors())
    return await request_validation_exception_handler(request=request, exc=exc)


async def update_uvicorn_logger() -> None:
    """Log all uvicorn errors."""
    logger_uvicorn = logging.getLogger("uvicorn.access")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger_uvicorn.addHandler(handler)


@app.get("/doc")
async def api_doc_redirect(request: Request) -> Response:
    """Redirect to API doc page."""
    return RedirectResponse("/api/doc")


# Root endpoint
@app.get("/")
async def root():
    """Home page."""
    return {"message": "Welcome to the Fraud Detection API"}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check api health status."""
    return {"status": "healthy"}
