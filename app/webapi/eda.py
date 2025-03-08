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


@router.post("/upload-file/", status_code=status.HTTP_201_CREATED)
async def perform_eda_endpoint(
    csv_file: UploadFile = File(...),
    request_user: User = Depends(user_auth.require([RoleEnum.ADMIN])),
) -> dict[str, str]:
    """
    Endpoint pour effectuer une analyse exploratoire des données (EDA) à partir d'un fichier CSV.

    Args:
        csv_file (UploadFile): Fichier CSV téléversé par l'utilisateur.
        request_user (User): Utilisateur authentifié avec le rôle ADMIN.

    Returns:
        dict[str, str]: Dictionnaire contenant les graphiques encodés en base64.

    Raises:
        HTTPException: Si une erreur survient lors de la lecture du fichier ou du traitement des données.
    """
    try:
        # Lire le contenu du fichier CSV
        contents = await csv_file.read()
    except Exception as e:
        logger.error("Erreur lors de la lecture du fichier: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la lecture du fichier",
        ) from e

    try:
        # Effectuer l'analyse exploratoire des données
        return eda.perform_eda(contents)
    except HTTPException:
        # Les exceptions FastAPI générées par perform_eda sont simplement propagées
        raise
    except Exception as e:
        logger.error("Erreur inattendue: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur inattendue lors du traitement: {str(e)}",
        ) from e
