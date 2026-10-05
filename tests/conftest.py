"""Configuration shared by all tests.

Settings are cached on first read, so the test environment is fixed here, before
any `src` module is imported. Connection details (host, port, owner and runtime
credentials) still come from `.env` or the environment.

Integration tests never use the application's database: POSTGRES_DB is forced to
`clinic_test` (or TEST_POSTGRES_DB), which the integration fixtures drop and
recreate at the start of every run.
"""

from __future__ import annotations

import os

os.environ["APP_ENV"] = "ci"
os.environ["ALLOW_REAL_PATIENT_DATA"] = "false"
# Readable placeholders, not secrets. Worded so they cannot resemble a provider
# API key format and trigger secret scanners.
os.environ["PHI_ENCRYPTION_KEY"] = "placeholder-test-phi-encryption-value"
os.environ["PHI_BLIND_INDEX_KEY"] = "placeholder-test-phi-blind-index-value"
os.environ["POSTGRES_DB"] = os.environ.get("TEST_POSTGRES_DB", "clinic_test")
if not os.environ["POSTGRES_DB"].endswith("_test"):
    raise RuntimeError("Integration tests may only use a database whose name ends in '_test'.")

# No test calls a live model provider (CLAUDE.md section 6). The keys are
# cleared here rather than trusted to be absent: a developer with a working key
# in `.env` would otherwise have every ambiguous column and every refusal make a
# real, paid, non-deterministic call, and the suite would pass differently on
# their machine than in CI. Tests that exercise these paths inject a fake client
# or a Settings object of their own.
os.environ["OPENAI_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""

from src.core.db import configure_event_loop_policy

configure_event_loop_policy()
