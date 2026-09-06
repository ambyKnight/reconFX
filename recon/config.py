"""
Configuration and constants for the reconciliation pipeline.

Per PIPELINE.md Part 4: every constant lives in this file and is hashed
into the run manifest for strict reproducibility.
"""

import hashlib
import json
import os
from pathlib import Path

# Dataset identifiers
DATASET_SLUG = "benchmarkteam/benchrec-real-world-cash-reconciliation-dataset"

# Blocking & windowing constants
DATE_WINDOW_DAYS = 5

# Benchmark evaluation constraint
REQUIRED_PRECISION = 0.998

# Collision rule parameters:
# match.py originally had COLLISION_LIMIT = 5 which demoted everything >= 5.
# PIPELINE.md §1.1 showed this demotes bulk sweeps / batch postings (12+ band).
# The corrected rule demotes only the risky middle band [COLLISION_DEMOTE_MIN, COLLISION_PROMOTE_MIN).
COLLISION_LIMIT = 5  # Kept for backwards compatibility as the demote entry threshold
COLLISION_DEMOTE_MIN = 5
COLLISION_PROMOTE_MIN = 12  # Promotes high-collision batch postings (fitted on train split)

# Confidence literals (ranking and calibration tiers)
CONF_SINGLE_ALLOCATION = 0.95
CONF_DATE_WINDOW_PENALTY = 0.10  # Deducted when matching on near-date window (-> 0.85)
CONF_REF_TOKEN_TIE_BREAK = 0.80
CONF_COLLISION_DEMOTED = 0.50
CONF_AMBIGUOUS = 0.35
CONF_NO_CANDIDATE = 0.0

# Suspense & materiality thresholds
WRITE_OFF_THRESHOLD = 25.00  # Configurable threshold for write-off proposal (PIPELINE §5.2, ARCHITECTURE §4.5)
CONF_AUTO_THRESHOLD = 0.95  # Calibrated threshold required for AUTO posting

# Reference tokenization pattern
TOKEN_REGEX = r"[A-Za-z0-9]{4,}"

# Path configuration
REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
FITTED_CONFIG_PATH = ARTIFACTS_DIR / "fitted_config.json"


def get_config_dict():
    """Return dictionary of current configuration parameters."""
    return {
        "dataset_slug": DATASET_SLUG,
        "date_window_days": DATE_WINDOW_DAYS,
        "required_precision": REQUIRED_PRECISION,
        "collision_limit": COLLISION_LIMIT,
        "collision_demote_min": COLLISION_DEMOTE_MIN,
        "collision_promote_min": COLLISION_PROMOTE_MIN,
        "conf_single_allocation": CONF_SINGLE_ALLOCATION,
        "conf_date_window_penalty": CONF_DATE_WINDOW_PENALTY,
        "conf_ref_token_tie_break": CONF_REF_TOKEN_TIE_BREAK,
        "conf_collision_demoted": CONF_COLLISION_DEMOTED,
        "conf_ambiguous": CONF_AMBIGUOUS,
        "conf_no_candidate": CONF_NO_CANDIDATE,
        "conf_auto_threshold": CONF_AUTO_THRESHOLD,
        "write_off_threshold": WRITE_OFF_THRESHOLD,
        "token_regex": TOKEN_REGEX,
    }


def get_config_hash():
    """Compute sha256 hash of configuration dictionary for the run manifest."""
    cfg = get_config_dict()
    encoded = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def save_fitted_config(params, path=None):
    """Save fitted parameters to JSON file in artifacts."""
    target_path = Path(path) if path else FITTED_CONFIG_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def load_fitted_config(path=None):
    """Load fitted parameters if available."""
    target_path = Path(path) if path else FITTED_CONFIG_PATH
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
