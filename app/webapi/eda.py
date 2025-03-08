"""
# WebAPI.

The API endpoint  is queried by other applications
to communicate with this application. This endpoint usually relies
on a header based authenticated encoded in the request headers.
Commonly Basic or Bearer Auth.

It should accept and returns data formatted in JSON.

The API is structured with  Representational state transfer architecture:
https://en.wikipedia.org/wiki/Representational_state_transfer
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.controllers import eda
from app.models import User
from app.models.enums import RoleEnum
from app.models.user.auth import user_auth

router = APIRouter()


logger = logging.getLogger("app")


@router.post("/", status_code=status.HTTP_201_CREATED)
async def perform_eda_endpoint(
    csv_file: UploadFile = File(...),
    request_user: User = Depends(user_auth.require([RoleEnum.INST])),
    batch_size: int = Query(100, description="Taille du lot pour le traitement des requêtes IP"),
    skip_ip_check: bool = Query(False, description="Ignorer la vérification des pays par IP (plus rapide)"),
) -> dict[str, str]:
    try:
        contents = await csv_file.read()
    except Exception as e:
        logger.error("Erreur lors de la lecture du fichier: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la lecture du fichier",
        ) from e

    try:
        return eda.perform_eda(contents, batch_size=batch_size, skip_ip_check=skip_ip_check)
    except HTTPException:
        # Les exceptions FastAPI générées par perform_eda sont simplement propagées
        raise
    except Exception as e:
        logger.error("Erreur inattendue: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur inattendue lors du traitement: {str(e)}",
        ) from e
