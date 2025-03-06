import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

from app.xlib.s3 import s3_upload
from AppMain.settings import AppSettings


async def prepare_and_upload_data(csv_file_path: str) -> dict[str, str]:
    """
    Prépare les données et les téléverse sur S3.

    Args:
        csv_file_path (str): Chemin du fichier CSV local.

    Returns:
        dict[str, str]: Liens S3 des ensembles de données (train, val, test).
    """
    # 1. Charger les données depuis le fichier CSV
    df = pd.read_csv(csv_file_path)

    # 2. Nettoyer les données
    df["Transaction Amount"].fillna(df["Transaction Amount"].median(), inplace=True)
    df["Customer Age"].fillna(df["Customer Age"].median(), inplace=True)
    df.drop_duplicates(inplace=True)

    # 3. Normaliser les données
    scaler = StandardScaler()
    df[["Transaction Amount", "Customer Age"]] = scaler.fit_transform(df[["Transaction Amount", "Customer Age"]])

    # 4. Diviser les données en ensembles d'entraînement, de validation et de test
    X = df.drop("Is Fraudulent", axis=1)
    y = df["Is Fraudulent"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 5. Sauvegarder les ensembles dans des fichiers CSV en mémoire
    train_set = pd.concat([X_train, y_train], axis=1)
    val_set = pd.concat([X_val, y_val], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    train_csv = train_set.to_csv(index=False)
    val_csv = val_set.to_csv(index=False)
    test_csv = test_set.to_csv(index=False)

    # 6. Téléverser les fichiers CSV sur S3
    project_name = AppSettings.PROJ_NAME

    train_key = f"{project_name}/train_set.csv"
    validation_key = f"{project_name}/val_set.csv"
    test_key = f"{project_name}/test_set.csv"

    train_url = await s3_upload(train_csv.encode(), train_key, AppSettings.AWS_S3_BUCKET)
    val_url = await s3_upload(val_csv.encode(), validation_key, AppSettings.AWS_S3_BUCKET)
    test_url = await s3_upload(test_csv.encode(), test_key, AppSettings.AWS_S3_BUCKET)

    return {
        "project_name": project_name,
        "train_set_url": train_url,
        "val_set_url": val_url,
        "test_set_url": test_url,
    }
