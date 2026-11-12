"""Synthetic seeding in every environment, never beside real patient data (ADR-11).

The PDF requires synthetic data generation for all environments. Local and CI seed
automatically; staging and production seed only on request; nothing seeds once
real patient data is enabled.
"""

from __future__ import annotations

import random

import phonenumbers
import pytest

from main import seeding_decision
from src.core.config import Settings
from src.devdata import seed
from src.devdata.seed import _colombian_mobile

STRONG = "s" * 40


def _settings(app_env: str, *, allow_real: bool = False) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        app_env=app_env,  # type: ignore[arg-type]
        allow_real_patient_data=allow_real,
        phi_encryption_key=STRONG,
        phi_blind_index_key=STRONG + "b",
        audit_chain_key=STRONG + "a",
        postgres_password="owner-secret",
        app_db_password="runtime-secret",
    )


@pytest.mark.parametrize("app_env", ["local", "ci", "staging", "production"])
def test_every_environment_may_seed_while_real_data_is_disabled(app_env: str) -> None:
    assert _settings(app_env).synthetic_seeding_allowed


@pytest.mark.parametrize("app_env", ["local", "ci", "staging", "production"])
def test_no_environment_may_seed_once_real_data_is_enabled(app_env: str) -> None:
    settings = _settings(app_env, allow_real=True)
    assert not settings.synthetic_seeding_allowed
    assert seeding_decision(settings, skip_seed=False, seed=True) == (
        "real patient data is enabled"
    )


@pytest.mark.parametrize("app_env", ["local", "ci"])
def test_local_and_ci_seed_automatically(app_env: str) -> None:
    assert seeding_decision(_settings(app_env), skip_seed=False, seed=False) is None


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_staging_and_production_seed_only_when_asked(app_env: str) -> None:
    settings = _settings(app_env)
    assert "pass --seed" in (seeding_decision(settings, skip_seed=False, seed=False) or "")
    assert seeding_decision(settings, skip_seed=False, seed=True) is None


def test_skip_seed_wins() -> None:
    assert seeding_decision(_settings("local"), skip_seed=True, seed=True) == "--skip-seed"


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_the_review_api_and_reset_stay_local_only(app_env: str) -> None:
    """Seeding widened; the unauthenticated review API and --reset-db did not."""
    assert not _settings(app_env).synthetic_data_mode


async def test_the_seeder_refuses_once_real_data_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seeder enforces the gate itself, not only main.py."""
    monkeypatch.setattr(seed, "get_settings", lambda: _settings("staging", allow_real=True))
    with pytest.raises(RuntimeError, match="never be mixed with real patient data"):
        await seed.run_seed()


def test_every_seeded_phone_number_is_a_real_colombian_mobile() -> None:
    """A seeded patient must be able to prove who they are.

    The generator was "+573 then nine random digits", which produces
    possible-looking invalid numbers: the 39x and 33x ranges are not assigned,
    so +573954336730 parses and fails validation. Nothing noticed until the
    patient-facing cancel route began validating the number it was handed --
    a seeded patient presenting their own number got "el número de teléfono no
    es válido", because it was not a real number.

    Checked against libphonenumber rather than a regex, for CLAUDE.md §11's
    reason: its metadata tracks the 2021 renumbering and a hand-rolled rule
    does not. That applies to generating a number as much as to checking one.

    Five thousand, not five: the defect was a minority of draws, so a handful
    of samples would have passed against the broken generator.
    """
    # Synthetic data, not crypto: the seed makes the failure reproducible.
    rng = random.Random(20261021)  # noqa: S311
    invalid = [
        number
        for number in (_colombian_mobile(rng) for _ in range(5000))
        if not phonenumbers.is_valid_number(phonenumbers.parse(number, "CO"))
    ]
    assert not invalid, f"{len(invalid)} of 5000 seeded numbers are not valid: {invalid[:5]}"
