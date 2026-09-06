"""
Tests for configuration constants, hashing, and fitted config persistence.
"""

from pathlib import Path
import tempfile
import pytest

from recon.config import (
    COLLISION_DEMOTE_MIN,
    COLLISION_LIMIT,
    COLLISION_PROMOTE_MIN,
    CONF_AMBIGUOUS,
    CONF_COLLISION_DEMOTED,
    CONF_NO_CANDIDATE,
    CONF_REF_TOKEN_TIE_BREAK,
    CONF_SINGLE_ALLOCATION,
    DATE_WINDOW_DAYS,
    REQUIRED_PRECISION,
    get_config_dict,
    get_config_hash,
    load_fitted_config,
    save_fitted_config,
)


def test_config_constants():
    assert DATE_WINDOW_DAYS == 5
    assert COLLISION_LIMIT == 5
    assert COLLISION_DEMOTE_MIN == 5
    assert COLLISION_PROMOTE_MIN >= 10
    assert REQUIRED_PRECISION == 0.998
    assert CONF_SINGLE_ALLOCATION == 0.95
    assert CONF_REF_TOKEN_TIE_BREAK == 0.80
    assert CONF_COLLISION_DEMOTED == 0.50
    assert CONF_AMBIGUOUS == 0.35
    assert CONF_NO_CANDIDATE == 0.0


def test_config_hash():
    h1 = get_config_hash()
    assert isinstance(h1, str)
    assert len(h1) == 16
    h2 = get_config_hash()
    assert h1 == h2


def test_save_and_load_fitted_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_fitted.json"
        params = {"collision_demote_min": 5, "collision_promote_min": 12, "date_window_days": 5}
        save_fitted_config(params, path=tmp_path)
        loaded = load_fitted_config(path=tmp_path)
        assert loaded == params
