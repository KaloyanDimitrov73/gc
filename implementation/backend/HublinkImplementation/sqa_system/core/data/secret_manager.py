from enum import Enum
import json
import os
from pathlib import Path
from typing import List, Tuple
from cryptography.fernet import Fernet
from sqa_system.core.language_model.enums.llm_enums import EndpointType


class SecretType(Enum):
    """Enum for the type of secrets that can be stored in the secret manager."""
    API_KEY = "api_key"
    PWD = "password"
    EMAIL = "email"


class SecretManager:
    """
    A class that manages secrets such as API keys and passwords for the project.

    It stores the secrets in a JSON file that is encrypted using the Fernet 
    symmetric encryption algorithm: https://cryptography.io/en/latest/fernet/ 

    The File is stored in the users home directory.

    Args: 
        file_name (str): The name of the file to store the secrets in.
    """

    def __init__(self, file_name="secrets.json"):
        config_dir_path = os.path.join(Path.home(), ".qa_system_master_thesis")
        self.config_file_path = os.path.join(config_dir_path, file_name)
        self.key_file_path = os.path.join(config_dir_path, ".key")
        self._ensure_dir_exists(path=config_dir_path)
        self._load_or_create_key()

    def _ensure_dir_exists(self, path):
        os.makedirs(path, exist_ok=True)

    def _load_or_create_key(self):
        if not os.path.exists(self.key_file_path):
            self.key = Fernet.generate_key()
            with open(self.key_file_path, "wb") as key_file:
                key_file.write(self.key)
        else:
            with open(self.key_file_path, "rb") as key_file:
                self.key = key_file.read()
        self.cipher_suite = Fernet(self.key)

    def get_all_secret_ids(self) -> List[Tuple[str, str]]:
        """
        Returns a dictionary of all secret identifiers and their corresponding secret type.

        Returns:
            dict[str, str]: A dictionary of all secret identifiers and their 
            corresponding secret type.
        """
        secret_ids = []
        secrets = self.load_secrets()
        for secret_type, identifiers in secrets.items():
            for identifier in identifiers:
                secret_ids.append((identifier, secret_type))
        return secret_ids

    def save_secret(self, secret_type: SecretType, identifier: str, secret: str):
        """
        Saves a secret for a given type and identifier.

        Args:
            secret_type (SecretType): The type of secret.
            identifier (str or EndpointType): The identifier for the secret.
            secret (str): The secret to be saved.
        """
        secrets = self.load_secrets()
        if secret_type.value not in secrets:
            secrets[secret_type.value] = {}
        secrets[secret_type.value][str(identifier)] = secret
        encrypted_secrets = self.cipher_suite.encrypt(
            json.dumps(secrets).encode())
        with open(self.config_file_path, "wb") as f:
            f.write(encrypted_secrets)

    def load_secrets(self):
        """
        Load the secrets from the configuration file.

        Returns:
            dict: A dictionary containing the secrets.
        """
        if not os.path.exists(self.config_file_path):
            return {}
        with open(self.config_file_path, "rb") as f:
            encrypted_secrets = f.read()
        decrypted_secrets = self.cipher_suite.decrypt(encrypted_secrets)
        return json.loads(decrypted_secrets)

    def get_secret(self,
                   secret_type: SecretType,
                   identifier: str,
                   prompt_user_if_failed: bool = True) -> str:
        """
        Retrieves a secret for a given type and identifier.

        Args:
            secret_type (SecretType): The type of secret .
            identifier (str or EndpointType): The identifier for the secret.
            prompt_user_if_failed (bool): Whether to prompt the user for the 
                secret if it is not found.

        Returns:
            str: The secret associated with the given type and identifier.

        Raises:
            ValueError: If no secret is found for the given type and identifier.
        """
        secrets = self.load_secrets()
        secret = secrets.get(secret_type.value, {}).get(str(identifier))
        if secret is None:
            if prompt_user_if_failed:
                secret = input(
                    f"Please enter the {secret_type.value} for {identifier}: ")
                self.save_secret(secret_type, identifier, secret)
            else:
                raise ValueError(
                    f"No {secret_type.value} found for identifier {identifier}")
        return secret

    def delete_secret(self, secret_type: SecretType, identifier: str) -> bool:
        """
        Deletes a secret for a given type and identifier.

        Args:
            secret_type (SecretType): The type of secret.
            identifier (str or EndpointType): The identifier for the secret.
            
        Returns:
            bool: True if the secret was deleted, False otherwise.
        """
        secrets = self.load_secrets()
        if secret_type.value in secrets and str(identifier) in secrets[secret_type.value]:
            del secrets[secret_type.value][str(identifier)]
            encrypted_secrets = self.cipher_suite.encrypt(
                json.dumps(secrets).encode())
            with open(self.config_file_path, "wb") as f:
                f.write(encrypted_secrets)
            return True
        return False

    # Convenience methods for API keys
    def save_api_key(self, endpoint: EndpointType, api_key: str):
        """
        Saves the API key for a given endpoint.

        Args:
            endpoint (EndpointType): The endpoint for which the API key is being saved.
            api_key (str): The API key to be saved.
        """
        self.save_secret(SecretType.API_KEY, endpoint.value, api_key)

    def get_api_key(self, endpoint: EndpointType) -> str:
        """
        Retrieves the API key for a given endpoint.

        Args:
            endpoint (EndpointType): The endpoint for which the API key is being retrieved.

        Returns:
            str: The API key associated with the given endpoint.
        """
        return self.get_secret(SecretType.API_KEY, endpoint.value)

    def delete_api_key(self, endpoint: EndpointType):
        """
        Deletes the API key for a given endpoint.

        Args:
            endpoint (EndpointType): The endpoint for which the API key is being deleted.
        """
        self.delete_secret(SecretType.API_KEY, endpoint.value)

    # Convenience methods for passwords
    def save_password(self, identifier: str, password: str):
        """
        Save a password for a given identifier.

        Args:
            identifier (str): The identifier for the password.
            password (str): The password to be saved.
        """
        self.save_secret(SecretType.PWD, identifier, password)

    def get_password(self, identifier: str) -> str:
        """
        Retrieves the password associated with the given identifier.

        Args:
            identifier (str): The identifier for the password.

        Returns:
            str: The password associated with the given identifier.
        """
        return self.get_secret(SecretType.PWD, identifier)

    def delete_password(self, identifier: str):
        """
        Deletes a password for a given identifier.

        Args:
            identifier (str): The identifier for the password.
        """
        self.delete_secret(SecretType.PWD, identifier)

    def save_email(self, identifier: str, email: str):
        """
        Save an email for a given identifier.

        Args:
            identifier (str): The identifier for the email.
            email (str): The email to be saved.
        """
        self.save_secret(SecretType.EMAIL, identifier, email)

    def get_email(self, identifier: str) -> str:
        """
        Retrieves the email associated with the given identifier.

        Args:
            identifier (str): The identifier for the email.

        Returns:
            str: The email associated with the given identifier.
        """
        return self.get_secret(SecretType.EMAIL, identifier)

    def delete_email(self, identifier: str):
        """
        Deletes an email for a given identifier.

        Args:
            identifier (str): The identifier for the email.
        """
        self.delete_secret(SecretType.EMAIL, identifier)
