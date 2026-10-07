"""Application settings.

Values are resolved by pydantic-settings in this order, highest priority first:

1. Process environment variables.
2. The `.env` file in the working directory.
3. Secret files in the directory named by the `SECRETS_DIR` environment
   variable: one file per setting, named after the field (for example
   `postgres_password`). Docker secrets, the Kubernetes Secrets Store CSI
   driver and Vault Agent all deliver secrets in this form, so a secrets
   manager can supply credentials without code changes (ADR-10).
4. The defaults declared below.

Two database identities are used deliberately:

* `postgres_*` is the owner account. It creates the database, runs migrations
  and owns the schema. It is used only by `main.py` and Alembic.
* `app_db_*` is the runtime account the API and the seeder connect as.
  `bootstrap.ensure_runtime_role` creates it. The initial migration grants it
  SELECT/INSERT/UPDATE/DELETE on `app` tables and only SELECT/INSERT on the
  audit log; `ensure_checkpoint_schema` grants it access to checkpoint tables.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from psycopg.conninfo import make_conninfo
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

AppEnv = Literal["local", "ci", "staging", "production"]

_PLACEHOLDER_PREFIXES = ("local-dev-", "placeholder-", "test-", "ci-", "change-me")
_MIN_KEY_LENGTH = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: AppEnv = "local"
    log_level: str = "INFO"

    # Owner account: database creation and migrations only.
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "clinic"
    postgres_user: str = "postgres"
    postgres_password: SecretStr = SecretStr("")

    # Runtime account used by the API and the seeder.
    app_db_user: str = "clinic_app"
    app_db_password: SecretStr = SecretStr("")

    # Column encryption and blind-index keys. See src/core/crypto.py.
    phi_encryption_key: SecretStr = SecretStr("")
    phi_blind_index_key: SecretStr = SecretStr("")

    # Keys the audit chain. Whoever holds it can forge a chain, so in staging and
    # production it belongs to a different secret store than the database
    # password: an attacker with database write access must not also hold it.
    audit_chain_key: SecretStr = SecretStr("")

    # ADR-11 / ADR-15 gate. While false, only synthetic data may be processed.
    allow_real_patient_data: bool = False

    rate_limit_per_minute: int = Field(default=60, ge=1)

    # How long a conversation's checkpoints are kept. They hold patient data, so
    # the retention sweep deletes them once a conversation has been idle this
    # long (see src/conversation/checkpointer.py).
    checkpoint_retention_days: int = Field(default=30, ge=1)

    # Column mapping may ask a model about a column the dictionary and fuzzy
    # matcher could not resolve. Both keys are optional: with neither set the
    # importer still works, and those columns go to the human instead. Nothing
    # here ever carries a patient's value — see src/onboarding/llm.py.
    openai_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    # A small, cheap model is right for this: the task is picking one of about
    # a dozen candidate fields, not reasoning.
    mapping_model_openai: str = "gpt-5-mini"
    mapping_model_groq: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Per call. A mapping proposal that takes longer than this is not worth
    # waiting for inside a request a person is watching.
    mapping_timeout_seconds: float = Field(default=30.0, gt=0)

    @property
    def mapping_llm_available(self) -> bool:
        """Whether a model can be asked about an ambiguous column at all."""
        return bool(self.openai_api_key.get_secret_value() or self.groq_api_key.get_secret_value())

    @property
    def synthetic_data_mode(self) -> bool:
        """True in local and CI while real patient data is disabled.

        Enables what is only safe on a developer's machine or a CI runner: the
        unauthenticated review API and `--reset-db`. Synthetic seeding is broader;
        see `synthetic_seeding_allowed`.
        """
        return self.app_env in ("local", "ci") and not self.allow_real_patient_data

    @property
    def synthetic_seeding_allowed(self) -> bool:
        """True in every environment until real patient data is enabled (ADR-11).

        Until the compliance gate is signed, staging and production hold only
        synthetic data too, so they must be seedable. Once real data is enabled,
        synthetic records would be mixed with real ones, so seeding stops.
        """
        return not self.allow_real_patient_data

    @property
    def admin_database_url(self) -> URL:
        return self._url(self.postgres_user, self.postgres_password)

    @property
    def app_database_url(self) -> URL:
        return self._url(self.app_db_user, self.app_db_password)

    def conninfo(self, *, admin: bool, dbname: str | None = None) -> str:
        """A libpq connection string for a direct psycopg connection."""
        user, password = (
            (self.postgres_user, self.postgres_password)
            if admin
            else (self.app_db_user, self.app_db_password)
        )
        return make_conninfo(
            host=self.postgres_host,
            port=self.postgres_port,
            dbname=dbname or self.postgres_db,
            user=user,
            password=password.get_secret_value(),
        )

    def _url(self, user: str, password: SecretStr) -> URL:
        # URL.create escapes special characters in the password, which string
        # formatting would not.
        return URL.create(
            drivername="postgresql+psycopg",
            username=user,
            password=password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )

    @model_validator(mode="after")
    def _require_real_secrets_outside_local(self) -> Settings:
        """Refuse to start staging or production on missing or placeholder secrets."""
        if self.app_env not in ("staging", "production"):
            return self
        for name in ("phi_encryption_key", "phi_blind_index_key", "audit_chain_key"):
            value = getattr(self, name).get_secret_value()
            if len(value) < _MIN_KEY_LENGTH or value.startswith(_PLACEHOLDER_PREFIXES):
                raise ValueError(
                    f"{name} must be a random secret of at least {_MIN_KEY_LENGTH} characters "
                    f"supplied by the secrets manager when APP_ENV={self.app_env}."
                )
        for name in ("postgres_password", "app_db_password"):
            value = getattr(self, name).get_secret_value()
            if not value or value.startswith(_PLACEHOLDER_PREFIXES):
                raise ValueError(
                    f"{name} is missing or a placeholder while APP_ENV={self.app_env}."
                )
        return self


@lru_cache
def get_settings() -> Settings:
    secrets_dir = os.environ.get("SECRETS_DIR")
    if secrets_dir and Path(secrets_dir).is_dir():
        # `_secrets_dir` is a pydantic-settings init option that mypy cannot see.
        return Settings(_secrets_dir=secrets_dir)  # type: ignore[call-arg]
    return Settings()
