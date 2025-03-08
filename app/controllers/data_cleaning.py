import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from app.xlib.s3 import s3_upload
from AppMain.settings import AppSettings


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données en remplaçant les valeurs manquantes et en supprimant les doublons.

    Args:
        df (pd.DataFrame): DataFrame contenant les données brutes.

    Returns:
        pd.DataFrame: DataFrame nettoyé.
    """
    df["Transaction Amount"].fillna(df["Transaction Amount"].median(), inplace=True)
    df["Customer Age"].fillna(df["Customer Age"].median(), inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les colonnes numériques du DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant les données nettoyées.

    Returns:
        pd.DataFrame: DataFrame normalisé.
    """
    scaler = StandardScaler()
    df[["Transaction Amount", "Customer Age"]] = scaler.fit_transform(df[["Transaction Amount", "Customer Age"]])
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divise les données en ensembles d'entraînement, de validation et de test.

    Args:
        df (pd.DataFrame): DataFrame contenant les données nettoyées et normalisées.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Ensembles d'entraînement, de validation et de test.
    """
    features = df.drop("Is Fraudulent", axis=1)
    target = df["Is Fraudulent"]

    features_train, features_temp, target_train, target_temp = train_test_split(
        features, target, test_size=0.3, random_state=42
    )
    features_val, features_test, target_val, target_test = train_test_split(
        features_temp, target_temp, test_size=0.5, random_state=42
    )

    train_set = pd.concat([features_train, target_train], axis=1)
    val_set = pd.concat([features_val, target_val], axis=1)
    test_set = pd.concat([features_test, target_test], axis=1)

    return train_set, val_set, test_set


async def upload_to_s3(train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame) -> dict[str, str]:
    """
    Téléverse les ensembles de données sur S3 et retourne les URLs.

    Args:
        train_set (pd.DataFrame): Ensemble d'entraînement.
        val_set (pd.DataFrame): Ensemble de validation.
        test_set (pd.DataFrame): Ensemble de test.

    Returns:
        dict[str, str]: Liens S3 des ensembles de données (train, val, test).

    Raises:
        ValueError: Si le téléversement sur S3 échoue.
    """
    train_csv = train_set.to_csv(index=False)
    val_csv = val_set.to_csv(index=False)
    test_csv = test_set.to_csv(index=False)

    project_name = AppSettings.PROJ_NAME

    train_key = f"{project_name}/train_set.csv"
    validation_key = f"{project_name}/val_set.csv"
    test_key = f"{project_name}/test_set.csv"

    train_url = await s3_upload(train_csv.encode(), train_key, AppSettings.AWS_S3_BUCKET)
    val_url = await s3_upload(val_csv.encode(), validation_key, AppSettings.AWS_S3_BUCKET)
    test_url = await s3_upload(test_csv.encode(), test_key, AppSettings.AWS_S3_BUCKET)

    if train_url is None or val_url is None or test_url is None:
        raise ValueError("Erreur lors du téléversement des fichiers sur S3.")

    return {
        "project_name": project_name,
        "train_set_url": train_url,
        "val_set_url": val_url,
        "test_set_url": test_url,
    }


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
    df = clean_data(df)

    # 3. Normaliser les données
    df = normalize_data(df)

    # 4. Diviser les données en ensembles d'entraînement, de validation et de test
    train_set, val_set, test_set = split_data(df)

    # 5. Téléverser les fichiers CSV sur S3
    return await upload_to_s3(train_set, val_set, test_set)
