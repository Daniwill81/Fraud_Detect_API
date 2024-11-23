from fastapi import HTTPException, UploadFile
import pandas as pd
from io import StringIO
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


# Fonction de chargement des données depuis le fichier CSV uploadé
def load_data_from_csv(file: UploadFile):
    try:
        # Lire le fichier CSV avec pandas
        csv_data = StringIO(file.read().decode('utf-8'))
        df = pd.read_csv(csv_data)
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la lecture du fichier CSV: {str(e)}")

# Fonction de prétraitement des données
def preprocess_data(df):
    X = df.drop("label", axis=1)  # Supposons que la colonne "label" contienne les étiquettes (fraude ou non)
    y = df["label"]
    
    # Séparer les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normaliser les données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Création du modèle MLP
def create_mlp_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # Sortie pour classification binaire

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Fonction d'entraînement du modèle
def train_model(X_train, y_train, X_test, y_test):
    model = create_mlp_model(X_train.shape[1])
    
    # Entraînement du modèle
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Évaluer le modèle
    loss, accuracy = model.evaluate(X_test, y_test)
    
    # Générer un rapport de classification
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    report = classification_report(y_test, y_pred)
    
    return model, accuracy, report
