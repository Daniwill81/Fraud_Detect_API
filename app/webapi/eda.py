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

from io import StringIO

import controllers
import controllers.eda
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()


@router.post("/perform-eda/")
async def perform_eda_endpoint(csv_file: UploadFile = File(...)) -> dict[str, str]:
    """
    Endpoint pour effectuer l'analyse exploratoire des données (EDA) sur un fichier CSV.

    Args:
        csv_file (UploadFile): Fichier CSV téléversé par l'utilisateur.

    Returns:
        dict[str, str]: Dictionnaire contenant les graphiques encodés en base64.
    """
    # Vérifier que le fichier est un CSV
    if not csv_file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Le fichier doit être au format CSV.")

    # Lire le fichier CSV dans un DataFrame
    try:
        contents = await csv_file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la lecture du fichier CSV : {str(e)}")

    # Effectuer l'EDA
    try:
        results = controllers.eda.perform_eda(df)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse des données : {str(e)}")
