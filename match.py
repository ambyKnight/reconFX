"""
Backward-compatible wrapper for recon reconciliation pipeline.

Per PIPELINE.md Part 4:
The core implementation is modularized under recon/ with constants in recon.config.
This entry point preserves the original API and command execution.

    python match.py
"""

from recon.blocking import collision_counts, index_ledger, measure_candidate_recall
from recon.config import (
    COLLISION_DEMOTE_MIN,
    COLLISION_LIMIT,
    COLLISION_PROMOTE_MIN,
    DATASET_SLUG as SLUG,
    DATE_WINDOW_DAYS,
    REQUIRED_PRECISION,
    TOKEN_REGEX,
)
from recon.decide import choose, match as recon_match
from recon.evaluate import evaluate as recon_evaluate
from recon.ingest import TOKEN_COMPILED as TOKEN, load_eval


def load():
    """Load evaluation dataset and ground truth solution."""
    ledger, bank, truth, n_total, _manifest = load_eval()
    return ledger, bank, truth, n_total


def match(bank, by_amount):
    """Run tiered matching with collision band promotion."""
    return recon_match(bank, by_amount)


def report(pred, truth, n_total, candidate_recall_stats=None):
    """Report evaluation coverage, tier breakdown, and confidence frontier."""
    evaluated_df, _metrics = recon_evaluate(
        pred, truth, n_total, candidate_recall_stats=candidate_recall_stats, print_report=True
    )
    return evaluated_df


if __name__ == "__main__":
    ledger, bank, truth, n_total = load()
    print(f"ingested {len(ledger)} ledger rows, {len(bank)} bank rows\n")
    by_amount = index_ledger(ledger)
    recall_stats = measure_candidate_recall(bank, by_amount, truth)
    pred = report(match(bank, by_amount), truth, n_total, candidate_recall_stats=recall_stats)
    pred.to_csv("predictions.csv", index=False)
    print("\nwrote predictions.csv")
