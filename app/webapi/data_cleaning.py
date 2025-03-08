"""
# WebAPI.

The API endpoint is queried by other applications
to communicate with this application. This endpoint usually relies
on a header-based authentication encoded in the request headers.
Commonly Basic or Bearer Auth.

It should accept and return data formatted in JSON.

The API is structured with Representational State Transfer architecture:
https://en.wikipedia.org/wiki/Representational_state_transfer
"""

import os  # Importation déplacée en haut du fichier

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.controllers import data_cleaning
from app.models import User
from app.models.enums import RoleEnum
from app.models.user.auth import user_auth
from AppMain.settings import AppSettings

router = APIRouter()


@router.post("/", status_code=status.HTTP_201_CREATED)
async def prepare_and_upload_data_endpoint(
    csv_file: UploadFile = File(...),
    request_user: User = Depends(user_auth.require([RoleEnum.INST])),
) -> dict[str, str]:
    """
    Endpoint pour préparer les données et les téléverser sur S3.

    Args:
        csv_file: Fichier CSV téléversé par l'utilisateur.
        request_user: Utilisateur authentifié.

    Returns:
        Liens S3 des ensembles de données (train, val, test).
    """
    # Garantir que csv_file.filename n'est pas None
    assert csv_file.filename is not None, "Le fichier doit avoir un nom."

    # Vérifier que le fichier se termine par .csv
    if not csv_file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Le fichier doit être au format CSV.")

    # Sauvegarder temporairement le fichier CSV
    temp_file_path = os.path.join(AppSettings.LOG_DIR, csv_file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await csv_file.read())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'écriture du fichier temporaire : {str(e)}",
        ) from e

    try:
        # Préparer les données et les téléverser sur S3
        result = await data_cleaning.prepare_and_upload_data(temp_file_path)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la préparation des données : {str(e)}",
        ) from e
    finally:
        # Supprimer le fichier temporaire
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
