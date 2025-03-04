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

import controllers
import controllers.data_cleaning
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from AppMain.settings import AppSettings

router = APIRouter()


@router.post("/prepare-and-upload-data/")
async def prepare_and_upload_data_endpoint(
    csv_file: UploadFile = File(...),
    request_user: User = Depends(user_auth.require([RoleEnum.SA1])),
) -> dict[str, str]:
    """
    Endpoint pour préparer les données et les téléverser sur S3.

    Args:
        csv_file (UploadFile): Fichier CSV téléversé par l'utilisateur.
        request_user (User): Utilisateur authentifié.

    Returns:
        dict[str, str]: Liens S3 des ensembles de données (train, val, test).
    """
    # Vérifier que le fichier est un CSV
    if not csv_file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Le fichier doit être au format CSV.")

    # Sauvegarder temporairement le fichier CSV
    import os  # pylint: disable=import-outside-toplevel

    temp_file_path = os.path.join(AppSettings.LOG_DIR, csv_file.filename)
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await csv_file.read())

    try:
        # Préparer les données et les téléverser sur S3
        result = await controllers.data_cleaning.prepare_and_upload_data(temp_file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la préparation des données : {str(e)}")
    finally:
        # Supprimer le fichier temporaire
        import os

        os.remove(temp_file_path)
