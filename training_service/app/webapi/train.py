from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.trainning_deep_NN_model import load_data_from_csv

router = APIRouter()

# Endpoint pour uploader le fichier et entraîner le modèle
@app.post("/train")
async def train_fraud_detection_model(file: UploadFile = File(...)):
    try:
        # Charger les données depuis le fichier CSV
        df = load_data_from_csv(file)
        
        # Prétraiter les données
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Entraîner le modèle
        model, accuracy, report = train_model(X_train, y_train, X_test, y_test)
        
        # Sauvegarder le modèle entraîné
        model.save("fraud_detection_model.h5")
        
        return {
            "message": "Modèle entraîné avec succès",
            "accuracy": accuracy,
            "classification_report": report
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
