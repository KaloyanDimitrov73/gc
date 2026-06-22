"""
Setup utilities for configuring HubLink backend credentials and settings.
Setup for ORKG credentials and API keys.
"""
from typing import Optional

from backend.app.core.config import Settings, get_settings
from core.data.secret_manager import SecretManager, SecretType
from language_model.enums.llm_enums import EndpointType


class SetupManager:
    """
    Manager for setup and configuration of HubLink backend.

    Handles:
    - ORKG credential configuration
    - Credential validation and status checking
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.secret_manager = SecretManager()
        self.settings = settings or get_settings()

    def set_up_orkg_credentials(self):
        """
        Set up ORKG credentials in the SecretManager.

        Args:
            email: ORKG user email
            password: ORKG user password
        """
        email = self.settings.orkg_email
        password = self.settings.orkg_password
        if not email or not password:
            raise EnvironmentError(
                "ORKG credentials not found in settings. "
                "Please set ORKG_EMAIL and ORKG_PASSWORD."
            )

        # sqa_system expects the email under "orkg_mail".
        # Keep "orkg_email" too for backwards compatibility with older code.
        self.secret_manager.save_secret(SecretType.EMAIL, "orkg_email", email)
        self.secret_manager.save_secret(SecretType.PWD, "orkg_password", password)

    def set_up_vdl_api_key(self):
        """
        Set up VDL API key in the SecretManager to avoid interactive prompts.
        """
        api_key = self.settings.vdl_api_key or self.settings.openai_api_key
        if not api_key:
            raise EnvironmentError(
                "VDL API key not found in settings. "
                "Please set VDL_API_KEY (or OPENAI_API_KEY as fallback)."
            )

        # VDL endpoint is used by the default HubLink config.
        self.secret_manager.save_api_key(EndpointType.VDL, api_key)
