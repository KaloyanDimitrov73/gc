"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

_CORE_DIR = Path(__file__).resolve().parent
_APP_DIR = _CORE_DIR.parent
_BACKEND_DIR = _APP_DIR.parent
_RETRIEVAL_CONFIG_DIR = _APP_DIR / "modules" / "qa" / "configs"


class Settings(BaseSettings):
    # FastAPI/runtime
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    expose_docs: bool = False
    log_level: str = "INFO"
    trusted_hosts: str = "localhost,127.0.0.1,::1"
    suppress_healthcheck_access_log: bool = True
    suppress_orkg_paged_false_warning: bool = True

    # Request hardening
    max_request_body_bytes: int = 1_048_576

    # Retrieval/LLM
    answer_llm_model: Optional[str] = None
    retrieval_kg_config_path: str = str(
        _RETRIEVAL_CONFIG_DIR
        / "default_hublink_config_deep_distributed.json"
    )
    retrieval_llm_configs_path: str = str(
        _RETRIEVAL_CONFIG_DIR / "llm_configs" / "llm_configs.json"
    )

    # Secrets/credentials
    orkg_email: Optional[str] = None
    orkg_password: Optional[str] = None
    vdl_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    stats_api_key: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
