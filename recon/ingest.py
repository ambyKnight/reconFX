"""
Stage 0: Ingestion and canonicalization.

Provides typed, trusted in-memory representations so no downstream stage
re-parses CSVs. Caches dataset location so the evaluation loop does not
depend on network connectivity.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess

import kagglehub
import pandas as pd

from recon.config import DATASET_SLUG, REPO_ROOT, TOKEN_REGEX, get_config_hash

TOKEN_COMPILED = re.compile(TOKEN_REGEX)


def get_dataset_dir():
    """Resolve the dataset directory, using local caches first before network.

    Checks:
      1. BENCHREC_DATA_DIR environment variable
      2. Local workspace cache (kagglehub_cache/)
      3. User's cached kagglehub downloads (~/.cache/kagglehub/datasets/...)
      4. kagglehub.dataset_download() fallback
    """
    env_dir = os.environ.get("BENCHREC_DATA_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    local_repo_cache = REPO_ROOT / "kagglehub_cache"
    if (local_repo_cache / "BenchRec_cash_v1.0_eval.csv").exists():
        return local_repo_cache

    user_cache = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / "benchmarkteam"
        / "benchrec-real-world-cash-reconciliation-dataset"
    )
    if user_cache.exists():
        versions_dir = user_cache / "versions"
        if versions_dir.exists():
            versions = sorted(list(versions_dir.glob("*")), key=lambda p: p.name)
            if versions:
                latest = versions[-1]
                if (latest / "BenchRec_cash_v1.0_eval.csv").exists():
                    return latest

    # Fallback to kagglehub download if not found locally
    downloaded_path = kagglehub.dataset_download(DATASET_SLUG)
    return Path(downloaded_path)


def tokenize_references(s):
    """Canonical tokenization of reference strings across all pipeline stages.

    Extracts uppercase alphanumeric sequences of length >= 4.
    """
    if pd.isna(s) or s is None:
        return set()
    return set(TOKEN_COMPILED.findall(str(s).upper()))


def parse_sides(df):
    """Split raw dataframe into typed ledger (A) and bank (B) frames."""
    ledger = df[df.A_id.notna()].copy()
    bank = df[df.B_id.notna()].copy()

    for sub_df, side in ((ledger, "A"), (bank, "B")):
        sub_df["amt"] = pd.to_numeric(sub_df[f"{side}_amount"], errors="coerce").round(2)
        sub_df["date"] = pd.to_datetime(sub_df[f"{side}_valueDate"], errors="coerce")
        sub_df["tokens"] = [
            tokenize_references(r) for r in sub_df[f"{side}_transactionReferences"].fillna("")
        ]

    return ledger, bank


def get_git_sha():
    """Retrieve current git commit SHA if inside a git repository."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def create_manifest(dataset_path=None, config_hash=None):
    """Emit a run manifest for reproducibility per Stage 0 spec."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "config_hash": config_hash or get_config_hash(),
        "dataset_path": str(dataset_path or get_dataset_dir()),
        "dataset_slug": DATASET_SLUG,
    }


def load_eval(data_dir=None):
    """Load and parse evaluation set and ground truth solution."""
    root = Path(data_dir) if data_dir else get_dataset_dir()
    eval_csv = root / "BenchRec_cash_v1.0_eval.csv"
    sol_csv = root / "BenchRec_cash_v1.0_solution.csv"

    ev = pd.read_csv(eval_csv, dtype=str, low_memory=False)
    sol = pd.read_csv(sol_csv, dtype=str)

    ledger, bank = parse_sides(ev)
    truth = dict(zip(sol.B_id, sol.targetAllocation))
    manifest = create_manifest(dataset_path=root)

    return ledger, bank, truth, len(sol), manifest


def load_train(data_dir=None):
    """Load and parse training set with ground truth target allocations."""
    root = Path(data_dir) if data_dir else get_dataset_dir()
    train_csv = root / "BenchRec_cash_v1.0_train.csv"

    tr = pd.read_csv(train_csv, dtype=str, low_memory=False)
    ledger, bank = parse_sides(tr)
    truth = dict(zip(bank.B_id, bank.targetAllocation))
    manifest = create_manifest(dataset_path=root)

    return ledger, bank, truth, len(bank), manifest
