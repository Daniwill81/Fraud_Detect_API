import os
import pickle
from typing import Optional, Union

import numpy as np
import tenseal as ts
from tenseal.enc_context import SCHEME_TYPE


class HomomorphicEncryptionService:
    """
    Service pour le chiffrement homomorphe utilisant TenSEAL avec CKKS.
    Permet d'effectuer des opérations sur des données chiffrées.
    """

    def __init__(
        self,
        scheme: SCHEME_TYPE = "CKKS",
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: Optional[list[int]] = None,
        security_level: int = 128,
        key_dir: str = "crypto_keys",
    ):
        """
        Initialise le service de chiffrement homomorphe.

        Args:
            poly_modulus_degree: Degré du polynôme modulaire, détermine la capacité des calculs.
            coeff_mod_bit_sizes: Tailles des bits pour les coefficients modulaires.
            security_level: Niveau de sécurité souhaité (128 ou 192 bits).
            key_dir: Répertoire pour stocker les clés.
        """
        self.key_dir = key_dir
        self.context_params = None
        self.context = None

        # Valeurs par défaut optimisées pour CKKS
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        # Création du contexte TenSEAL
        self._create_context(scheme, poly_modulus_degree, coeff_mod_bit_sizes, security_level)

        # Création des clés si elles n'existent pas
        if not os.path.exists(key_dir):
            os.makedirs(key_dir)
            self._generate_keys()
        elif not (os.path.exists(f"{key_dir}/public.key") and os.path.exists(f"{key_dir}/private.key")):
            self._generate_keys()

    def _create_context(
        self, scheme: SCHEME_TYPE, poly_modulus_degree: int, coeff_mod_bit_sizes: list[int], security_level: int
    ) -> None:
        """
        Crée le contexte CKKS.

        Args:
            poly_modulus_degree: Degré du polynôme modulaire.
            coeff_mod_bit_sizes: Tailles des bits pour les coefficients modulaires.
            security_level: Niveau de sécurité.
        """
        # Création des paramètres du contexte CKKS
        self.context = ts.Context(
            scheme=scheme,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            security_level=security_level,
        )

        # Configuration pour le chiffrement homomorphe
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40  # Échelle globale pour la précision

    def _generate_keys(self) -> None:
        """Génère et sauvegarde les clés publiques et privées."""
        # Sauvegarder la clé secrète
        secret_key = self.context.secret_key()
        with open(f"{self.key_dir}/private.key", "wb") as f:
            pickle.dump(secret_key, f)

        # Sauvegarder la clé publique (contexte sans clé privée)
        self.context.make_context_public()
        with open(f"{self.key_dir}/public.key", "wb") as f:
            pickle.dump(self.context, f)

        # Rétablir le contexte avec la clé privée
        self.context.secret_key(secret_key)

    def load_context(self, public_only: bool = False) -> ts.Context:
        """
        Charge le contexte depuis les fichiers de clés.

        Args:
            public_only: Si True, charge uniquement le contexte public.

        Returns:
            Le contexte TenSEAL chargé.
        """
        if public_only:
            with open(f"{self.key_dir}/public.key", "rb") as f:
                self.context = pickle.load(f)
        else:
            # Charger la clé privée
            with open(f"{self.key_dir}/private.key", "rb") as f:
                secret_key = pickle.load(f)

            # Charger le contexte public
            with open(f"{self.key_dir}/public.key", "rb") as f:
                self.context = pickle.load(f)

            # Restaurer la clé privée dans le contexte
            self.context.secret_key(secret_key)

        return self.context

    def encrypt(self, data: Union[float, list[float], np.ndarray]) -> ts.CKKSTensor:
        """
        Chiffre des données avec CKKS.

        Args:
            data: Donnée(s) à chiffrer (nombre flottant ou liste/tableau de flottants).

        Returns:
            Données chiffrées sous forme de tenseur CKKS.
        """
        if self.context is None:
            raise ValueError("Le contexte CKKS n'est pas initialisé.")

        # Conversion en liste si nécessaire
        if isinstance(data, np.ndarray):
            data = data.tolist()
        elif isinstance(data, (int, float)):
            data = [float(data)]
        elif isinstance(data, list):
            data = [float(x) for x in data]

        # Chiffrement avec TenSEAL
        return ts.ckks_tensor(self.context, data)

    def decrypt(self, encrypted_data: ts.CKKSTensor) -> Union[float, list[float]]:
        """
        Déchiffre des données CKKS.

        Args:
            encrypted_data: Données chiffrées (tenseur CKKS).

        Returns:
            Données déchiffrées sous forme de liste ou de nombre flottant.
        """
        if self.context is None:
            raise ValueError("Le contexte CKKS n'est pas initialisé.")

        # Vérification de la présence de la clé privée
        if not self.context.has_secret_key():
            raise ValueError("Impossible de déchiffrer sans clé privée.")

        # Déchiffrement
        result = encrypted_data.decrypt()

        # Retourne un seul nombre si la liste ne contient qu'un élément
        return result[0] if len(result) == 1 else result

    def add(self, a: ts.CKKSTensor, b: Union[ts.CKKSTensor, float, list[float]]) -> ts.CKKSTensor:
        """Additionne deux tenseurs chiffrés ou un tenseur chiffré et des données non chiffrées."""
        if isinstance(b, (float, int, list, np.ndarray)):
            b = self.encrypt(b)
        return a + b

    def sub(self, a: ts.CKKSTensor, b: Union[ts.CKKSTensor, float, list[float]]) -> ts.CKKSTensor:
        """Soustrait deux tenseurs chiffrés ou un tenseur chiffré et des données non chiffrées."""
        if isinstance(b, (float, int, list, np.ndarray)):
            b = self.encrypt(b)
        return a - b

    def mul(self, a: ts.CKKSTensor, b: Union[ts.CKKSTensor, float, list[float]]) -> ts.CKKSTensor:
        """Multiplie deux tenseurs chiffrés ou un tenseur chiffré et des données non chiffrées."""
        if isinstance(b, (float, int, list, np.ndarray)):
            b = self.encrypt(b)
        return a * b


# Instance par défaut pour une utilisation simplifiée
_default_service = None


def get_encryption_service(
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: Optional[list[int]] = None,
    security_level: int = 128,
    key_dir: str = "crypto_keys",
) -> HomomorphicEncryptionService:
    """
    Obtient l'instance du service de chiffrement homomorphe.

    Args:
        poly_modulus_degree: Degré du polynôme modulaire.
        coeff_mod_bit_sizes: Tailles des bits pour les coefficients modulaires.
        security_level: Niveau de sécurité (128 ou 192 bits).
        key_dir: Répertoire pour stocker les clés.

    Returns:
        Instance du service de chiffrement.
    """
    global _default_service

    if _default_service is None:
        _default_service = HomomorphicEncryptionService(
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            security_level=security_level,
            key_dir=key_dir,
        )

    return _default_service


def encrypt_data(data: Union[float, list[float], np.ndarray]) -> ts.CKKSTensor:
    """
    Fonction simplifiée pour chiffrer des données.

    Args:
        data: Données à chiffrer.

    Returns:
        Données chiffrées.
    """
    service = get_encryption_service()
    return service.encrypt(data)


def decrypt_data(encrypted_data: ts.CKKSTensor) -> Union[float, list[float]]:
    """
    Fonction simplifiée pour déchiffrer des données.

    Args:
        encrypted_data: Données chiffrées.

    Returns:
        Données déchiffrées.
    """
    service = get_encryption_service()
    return service.decrypt(encrypted_data)


# Exemple d'utilisation
# if __name__ == "__main__":
# Initialisation du service
#    service = get_encryption_service()

# Chiffrement de données
#    data = [1.5, 2.3, 3.7, 4.1]
#    encrypted = service.encrypt(data)
#    print(f"Données chiffrées: {encrypted}")

# Opération sur les données chiffrées
#    encrypted_result = service.add(encrypted, 2.0)

# Déchiffrement des résultats
#   result = service.decrypt(encrypted_result)
#   print(f"Résultat déchiffré: {result}")
#   print(f"Résultat attendu: {[x + 2.0 for x in data]}")
