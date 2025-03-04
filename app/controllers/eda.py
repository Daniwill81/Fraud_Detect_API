import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns


def get_country_from_ip(ip):
    """Fonction pour obtenir le pays à partir d'une adresse IP."""
    try:
        response = requests.get(f"https://ipinfo.io/{ip}/json")
        data = response.json()
        return data.get("country", "Unknown")
    except:
        return "Unknown"


def perform_eda(df: pd.DataFrame) -> dict[str, str]:
    """
    Effectue l'analyse exploratoire des données (EDA) et retourne les graphiques sous forme d'images encodées en base64.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.

    Returns:
        dict[str, str]: Dictionnaire contenant les graphiques encodés en base64.
    """
    # 1. Extraire le pays à partir des adresses IP
    df["IP Country"] = df["IP Address"].apply(get_country_from_ip)

    # 2. Créer une nouvelle colonne pour vérifier si l'adresse de livraison correspond à l'adresse de facturation
    df["Address_Match"] = (df["Shipping Address"] == df["Billing Address"]).astype(int)

    # 3. Encoder la colonne 'IP Country' en variables catégorielles
    df["IP Country"] = df["IP Country"].astype("category").cat.codes

    # 4. Générer les graphiques
    results = {}

    # Distribution de 'Transaction Amount' et 'Customer Age'
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df["Transaction Amount"], kde=True, color="blue")
    plt.title("Distribution de Transaction Amount")
    plt.subplot(1, 2, 2)
    sns.histplot(df["Customer Age"], kde=True, color="green")
    plt.title("Distribution de Customer Age")
    plt.tight_layout()
    results["distributions"] = plot_to_base64(plt)
    plt.close()

    # Matrice de corrélation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matrice de corrélation (incluant Address_Match et IP Country)")
    results["correlation_matrix"] = plot_to_base64(plt)
    plt.close()

    # Relation entre 'Transaction Amount' et 'Is Fraudulent'
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Is Fraudulent", y="Transaction Amount", data=df, palette="Set2")
    plt.title("Transaction Amount vs Is Fraudulent")
    results["transaction_amount_vs_fraud"] = plot_to_base64(plt)
    plt.close()

    # Nombre de transactions frauduleuses par méthode de paiement
    plt.figure(figsize=(10, 6))
    sns.countplot(x="Payment Method", hue="Is Fraudulent", data=df, palette="Set3")
    plt.title("Nombre de transactions frauduleuses par méthode de paiement")
    plt.xticks(rotation=45)
    results["fraud_by_payment_method"] = plot_to_base64(plt)
    plt.close()

    # Relation entre 'Address_Match' et 'Is Fraudulent'
    plt.figure(figsize=(8, 6))
    sns.countplot(x="Address_Match", hue="Is Fraudulent", data=df, palette="Set1")
    plt.title("Nombre de transactions frauduleuses par correspondance d'adresse")
    plt.xticks(ticks=[0, 1], labels=["Non", "Oui"])
    results["fraud_by_address_match"] = plot_to_base64(plt)
    plt.close()

    # Nombre de transactions frauduleuses par pays (IP Country)
    plt.figure(figsize=(12, 6))
    sns.countplot(x="IP Country", hue="Is Fraudulent", data=df, palette="Set2")
    plt.title("Nombre de transactions frauduleuses par pays (IP Country)")
    plt.xticks(rotation=90)
    results["fraud_by_ip_country"] = plot_to_base64(plt)
    plt.close()

    return results


def plot_to_base64(plt):
    """Convertit un graphique matplotlib en une image encodée en base64."""
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
