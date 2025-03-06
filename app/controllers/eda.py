import base64
import logging
import typing
from functools import lru_cache
from io import BytesIO, StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from fastapi import HTTPException, status

from AppMain.settings import AppSettings

# Configuration du logger
logger = logging.getLogger(__name__)


# Cache pour stocker les résultats des requêtes d'IP
@lru_cache(maxsize=1000)
def get_country_from_ip(ip: str) -> str:
    """
    Fonction pour obtenir le pays à partir d'une adresse IP avec mise en cache.

    Args:
        ip (str): L'adresse IP à vérifier.

    Returns:
        str: Le code pays ou "Unknown" en cas d'erreur.
    """
    if not ip or pd.isna(ip):
        return "Unknown"

    try:
        response = requests.get(f"{AppSettings.IP_INFO_URL}{ip}/", timeout=5)
        response.raise_for_status()  # Vérifie si la requête a réussi
        data = response.json()
        return data.get("country", "Unknown")
    except requests.exceptions.RequestException as e:
        # Gérer les erreurs de requête HTTP spécifiques
        logger.warning(f"Erreur lors de la récupération des informations d'IP {ip}: {e}")
        return "Unknown"
    except ValueError as e:
        # Gérer les erreurs de parsing JSON
        logger.warning(f"Erreur de format JSON pour l'IP {ip}: {e}")
        return "Unknown"
    except Exception as e:
        # Capture les autres exceptions inattendues
        logger.error(f"Erreur inattendue pour l'IP {ip}: {e}")
        return "Unknown"


def process_ip_countries_in_batches(df: pd.DataFrame, batch_size: int = 100) -> pd.Series:
    """
    Traite les adresses IP par lots pour éviter de surcharger l'API.

    Args:
        df (pd.DataFrame): DataFrame contenant une colonne 'IP Address'.
        batch_size (int): Taille du lot d'adresses IP à traiter à la fois.

    Returns:
        pd.Series: Série contenant les pays correspondant aux adresses IP.
    """
    results = pd.Series(index=df.index, dtype="object")

    # Traiter les adresses IP par lots
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        logger.info(f"Traitement du lot {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")

        # Appliquer la fonction de recherche de pays sur chaque IP du lot
        for idx, ip in zip(batch.index, batch["IP Address"]):
            results[idx] = get_country_from_ip(ip)

    return results


def perform_eda(df_or_file: bytes, batch_size: int = 100, skip_ip_check: bool = False) -> dict[str, str]:
    """
    Effectue l'analyse exploratoire des données (EDA) et retourne les graphiques sous forme d'images encodées en base64.
    Accepte soit un DataFrame, soit un fichier CSV.

    Args:
        df_or_file: DataFrame pandas ou contenu de fichier CSV.
        batch_size (int): Taille du lot pour le traitement des requêtes IP.
        skip_ip_check (bool): Ignorer la vérification des pays par IP pour accélérer le traitement.

    Returns:
        Dict[str, str]: Dictionnaire contenant les graphiques encodés en base64.
    """
    # Vérifier et charger les données
    if isinstance(df_or_file, pd.DataFrame):
        df = df_or_file
    else:
        try:
            # Si c'est un contenu de fichier, essayons de le charger
            df = pd.read_csv(StringIO(df_or_file.decode("utf-8")))
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Le fichier CSV n'est pas encodé en UTF-8."
            )
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier CSV: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erreur lors de la lecture du fichier CSV: {str(e)}",
            )

    # Vérifier les colonnes requises
    required_columns = [
        "IP Address",
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

    logger.info(f"Données chargées avec succès: {len(df)} lignes")

    # Créer une copie pour éviter de modifier le DataFrame original
    df_analysis = df.copy()

    # 1. Extraire le pays à partir des adresses IP (par lots ou ignorer)
    if skip_ip_check:
        logger.info("Vérification des pays par IP ignorée")
        df_analysis["IP Country"] = "Unknown"
        df_analysis["IP Country Original"] = "Unknown"
    else:
        logger.info("Extraction des pays à partir des adresses IP...")
        df_analysis["IP Country"] = process_ip_countries_in_batches(df_analysis, batch_size)
        # Garder une copie des pays d'origine pour le graphique
        df_analysis["IP Country Original"] = df_analysis["IP Country"]

    # 2. Créer une nouvelle colonne pour vérifier si l'adresse de livraison correspond à l'adresse de facturation
    df_analysis["Address_Match"] = (df_analysis["Shipping Address"] == df_analysis["Billing Address"]).astype(int)

    # 3. Pour les analyses numériques uniquement, encoder la colonne 'IP Country' en variables catégorielles
    ip_country_codes = df_analysis["IP Country"].astype("category").cat.codes
    df_analysis["IP Country Code"] = ip_country_codes

    # 4. Générer les graphiques
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

        # Matrice de corrélation (utiliser IP Country Code ici)
        numeric_cols = ["Transaction Amount", "Customer Age", "IP Country Code", "Address_Match", "Is Fraudulent"]
        corr_matrix = df_analysis[numeric_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice de corrélation (incluant Address_Match et IP Country)")
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

        # Nombre de transactions frauduleuses par pays (IP Country)
        if not skip_ip_check:
            # Utilisation des 10 pays les plus fréquents pour la lisibilité
            top_countries = df_analysis["IP Country Original"].value_counts().head(10).index.tolist()
            top_countries_df = df_analysis[df_analysis["IP Country Original"].isin(top_countries)].copy()

            plt.figure(figsize=(12, 6))
            sns.countplot(x="IP Country Original", hue="Is Fraudulent", data=top_countries_df, palette="Set2")
            plt.title("Nombre de transactions frauduleuses par pays (Top 10 pays)")
            plt.xticks(rotation=45)
            results["fraud_by_ip_country"] = plot_to_base64(plt)
            plt.close()

            # Taux de fraude par pays (nouveau graphique)
            fraud_by_country = (
                df_analysis.groupby("IP Country Original")["Is Fraudulent"].mean().sort_values(ascending=False).head(10)
            )
            plt.figure(figsize=(12, 6))
            fraud_by_country.plot(kind="bar", color="red")
            plt.title("Taux de fraude par pays (Top 10)")
            plt.ylabel("Taux de fraude")
            plt.xlabel("Pays")
            plt.xticks(rotation=45)
            results["fraud_rate_by_country"] = plot_to_base64(plt)
            plt.close()
    except Exception as e:
        logger.error(f"Erreur lors de la génération des graphiques: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la génération des graphiques: {str(e)}",
        )

    return results


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
