import base64
import logging
import typing
from io import BytesIO, StringIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fastapi import HTTPException, status

# Configuration du logger
logger = logging.getLogger(__name__)


def load_data(df_or_file: typing.Union[bytes, pd.DataFrame]) -> pd.DataFrame:
    """
    Charge les données à partir d'un DataFrame ou d'un fichier CSV.

    Args:
        df_or_file: DataFrame pandas ou contenu de fichier CSV.

    Returns:
        pd.DataFrame: DataFrame chargé.

    Raises:
        HTTPException: Si le fichier CSV n'est pas encodé en UTF-8 ou s'il y a une erreur de lecture.
    """
    if isinstance(df_or_file, pd.DataFrame):
        return df_or_file

    try:
        return pd.read_csv(StringIO(df_or_file.decode("utf-8")))
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Le fichier CSV n'est pas encodé en UTF-8."
        ) from exc
    except Exception as exc:
        logger.error("Erreur lors de la lecture du fichier CSV: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la lecture du fichier CSV: {str(exc)}",
        ) from exc


def generate_plots(df_analysis: pd.DataFrame) -> dict[str, str]:
    """
    Génère les graphiques pour l'analyse exploratoire des données.

    Args:
        df_analysis (pd.DataFrame): DataFrame contenant les données analysées.

    Returns:
        Dict[str, str]: Dictionnaire contenant les graphiques encodés en base64.

    Raises:
        HTTPException: Si une erreur survient lors de la génération des graphiques.
    """
    results = {}

    try:
        # Distribution de 'Transaction Amount' et 'Customer Age'
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df_analysis["Transaction Amount"], kde=True, color="blue")
        plt.title("Distribution de Transaction Amount")
        plt.subplot(1, 2, 2)
        sns.histplot(df_analysis["Customer Age"], kde=True, color="green")
        plt.title("Distribution de Customer Age")
        plt.tight_layout()
        results["distributions"] = plot_to_base64(plt)
        plt.close()

        # Matrice de corrélation
        numeric_cols = ["Transaction Amount", "Customer Age", "Address_Match", "Is Fraudulent"]
        corr_matrix = df_analysis[numeric_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice de corrélation (incluant Address_Match)")
        results["correlation_matrix"] = plot_to_base64(plt)
        plt.close()

        # Relation entre 'Transaction Amount' et 'Is Fraudulent'
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="Is Fraudulent", y="Transaction Amount", data=df_analysis, palette="Set2")
        plt.title("Transaction Amount vs Is Fraudulent")
        results["transaction_amount_vs_fraud"] = plot_to_base64(plt)
        plt.close()

        # Nombre de transactions frauduleuses par méthode de paiement
        plt.figure(figsize=(10, 6))
        sns.countplot(x="Payment Method", hue="Is Fraudulent", data=df_analysis, palette="Set3")
        plt.title("Nombre de transactions frauduleuses par méthode de paiement")
        plt.xticks(rotation=45)
        results["fraud_by_payment_method"] = plot_to_base64(plt)
        plt.close()

        # Relation entre 'Address_Match' et 'Is Fraudulent'
        plt.figure(figsize=(8, 6))
        sns.countplot(x="Address_Match", hue="Is Fraudulent", data=df_analysis, palette="Set1")
        plt.title("Nombre de transactions frauduleuses par correspondance d'adresse")
        plt.xticks(ticks=[0, 1], labels=["Non", "Oui"])
        results["fraud_by_address_match"] = plot_to_base64(plt)
        plt.close()

    except Exception as exc:
        logger.error("Erreur lors de la génération des graphiques: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la génération des graphiques: {str(exc)}",
        ) from exc

    return results


def perform_eda(df_or_file: typing.Union[bytes, pd.DataFrame]) -> dict[str, str]:
    """
    Effectue l'analyse exploratoire des données (EDA) et retourne les graphiques sous forme d'images encodées en base64.
    Accepte soit un DataFrame, soit un fichier CSV.

    Args:
        df_or_file: DataFrame pandas ou contenu de fichier CSV.

    Returns:
        Dict[str, str]: Dictionnaire contenant les graphiques encodés en base64.
    """
    # Charger les données
    df = load_data(df_or_file)

    # Vérifier les colonnes requises
    required_columns = [
        "Transaction Amount",
        "Customer Age",
        "Shipping Address",
        "Billing Address",
        "Payment Method",
        "Is Fraudulent",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Colonnes manquantes dans le CSV: {', '.join(missing_columns)}",
        )

    logger.info("Données chargées avec succès: %d lignes", len(df))

    # Créer une copie pour éviter de modifier le DataFrame original
    df_analysis = df.copy()

    # Créer une nouvelle colonne pour vérifier si l'adresse de livraison correspond à l'adresse de facturation
    df_analysis["Address_Match"] = (df_analysis["Shipping Address"] == df_analysis["Billing Address"]).astype(int)

    # Générer les graphiques
    return generate_plots(df_analysis)


def plot_to_base64(plt: typing.Any) -> str:
    """
    Convertit un graphique matplotlib en une image encodée en base64.

    Args:
        plt: Instance de matplotlib.pyplot

    Returns:
        str: Chaîne encodée en base64
    """
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
