"""
recon - End-to-end reconciliation package for BenchRec.

Per PIPELINE.md Part 4 & ARCHITECTURE.md:
Modular stages for cash reconciliation with calibrated confidence and candidate recall.
"""

from recon.blocking import (
    collision_counts,
    get_candidate_pool,
    index_ledger,
    measure_candidate_recall,
)
from recon.config import (
    COLLISION_DEMOTE_MIN,
    COLLISION_LIMIT,
    COLLISION_PROMOTE_MIN,
    CONF_AMBIGUOUS,
    CONF_AUTO_THRESHOLD,
    CONF_COLLISION_DEMOTED,
    CONF_NO_CANDIDATE,
    CONF_REF_TOKEN_TIE_BREAK,
    CONF_SINGLE_ALLOCATION,
    DATE_WINDOW_DAYS,
    REQUIRED_PRECISION,
    WRITE_OFF_THRESHOLD,
    get_config_dict,
    get_config_hash,
)
from recon.decide import choose, match
from recon.evaluate import evaluate
from recon.fit import fit_collision_thresholds
from recon.ingest import create_manifest, get_dataset_dir, load_eval, load_train, tokenize_references
from recon.journal import (
    Journal,
    get_default_journal,
    get_open_items,
    log_decision,
    reverse_decision,
)
from recon.suspense import (
    AGE_BUCKETS,
    CATEGORIES,
    SuspenseCaseFile,
    assess_materiality,
    build_suspense_register,
    categorize_suspense,
    create_case_file,
    get_age_bucket,
    reattempt_open_suspense,
)

__all__ = [
    "load_eval",
    "load_train",
    "get_dataset_dir",
    "tokenize_references",
    "create_manifest",
    "index_ledger",
    "collision_counts",
    "get_candidate_pool",
    "measure_candidate_recall",
    "choose",
    "match",
    "evaluate",
    "fit_collision_thresholds",
    "get_config_dict",
    "get_config_hash",
    "DATE_WINDOW_DAYS",
    "COLLISION_LIMIT",
    "COLLISION_DEMOTE_MIN",
    "COLLISION_PROMOTE_MIN",
    "REQUIRED_PRECISION",
    "CONF_SINGLE_ALLOCATION",
    "CONF_REF_TOKEN_TIE_BREAK",
    "CONF_COLLISION_DEMOTED",
    "CONF_AMBIGUOUS",
    "CONF_NO_CANDIDATE",
    "CONF_AUTO_THRESHOLD",
    "WRITE_OFF_THRESHOLD",
    "Journal",
    "get_default_journal",
    "log_decision",
    "reverse_decision",
    "get_open_items",
    "CATEGORIES",
    "AGE_BUCKETS",
    "SuspenseCaseFile",
    "categorize_suspense",
    "create_case_file",
    "get_age_bucket",
    "assess_materiality",
    "reattempt_open_suspense",
    "build_suspense_register",
]
