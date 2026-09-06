"""
Stage 1: Blocking (candidate generation) and candidate recall instrumentation.

Job: Shortlist potential counterparts cheaply before ranking or scoring.
Instruments the candidate recall ceiling (the fraction of items where the true
allocation is present anywhere in the blocking pool).
"""

from collections import defaultdict
import pandas as pd

from recon.config import DATE_WINDOW_DAYS


def index_ledger(ledger):
    """Index ledger rows by exact rounded amount."""
    by_amount = defaultdict(list)
    for row in ledger.itertuples():
        if pd.notna(row.amt):
            by_amount[row.amt].append(row)
    return by_amount


def collision_counts(bank):
    """Compute how many bank items share an item's exact (amt, date)."""
    return bank.groupby(["amt", "date"]).B_id.transform("size")


def get_candidate_pool(b, by_amount, date_window_days=DATE_WINDOW_DAYS):
    """Retrieve candidate ledger rows for a bank line.

    Returns:
      (pool, pool_type) where pool is same_day if available, else near within date_window_days.
    """
    exact = by_amount.get(b.amt, []) if pd.notna(b.amt) else []
    same_day = [a for a in exact if a.date == b.date]
    if same_day:
        return same_day, "same_day"

    near = [
        a for a in exact
        if pd.notna(a.date) and pd.notna(b.date)
        and abs((a.date - b.date).days) <= date_window_days
    ]
    return near, "near"


def parse_target_allocations(target_val):
    """Extract individual allocation strings from a ground truth target.

    Supports both single allocation strings and bracketed lists:
    '[USD_..., USD_...]'.
    """
    if not isinstance(target_val, str) or not target_val.strip():
        return []
    cleaned = target_val.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return [p.strip() for p in cleaned[1:-1].split(",") if p.strip()]
    return [cleaned]


def measure_candidate_recall(bank, by_amount, truth, date_window_days=DATE_WINDOW_DAYS):
    """Measure candidate recall: fraction of bank items whose true allocation is in the pool.

    This measures the blocking ceiling before any scoring or ranking takes place.
    """
    n_total = len(bank)
    n_with_target = 0
    pool_hits = 0
    window_hits = 0
    exact_hits = 0

    for b in bank.itertuples():
        t = truth.get(b.B_id)
        targets = parse_target_allocations(t)
        if not targets:
            continue

        n_with_target += 1
        target_set = set(targets)

        exact = by_amount.get(b.amt, []) if pd.notna(b.amt) else []
        same_day = [a for a in exact if a.date == b.date]
        near = [
            a for a in exact
            if pd.notna(a.date) and pd.notna(b.date)
            and abs((a.date - b.date).days) <= date_window_days
        ]
        pool = same_day or near

        pool_allocs = {a.A_allocation for a in pool}
        near_allocs = {a.A_allocation for a in near}
        exact_allocs = {a.A_allocation for a in exact}

        # Match if target string matches or any target set member is in candidate allocations
        if bool(target_set & pool_allocs) or (isinstance(t, str) and t.strip() in pool_allocs):
            pool_hits += 1
        if bool(target_set & near_allocs) or (isinstance(t, str) and t.strip() in near_allocs):
            window_hits += 1
        if bool(target_set & exact_allocs) or (isinstance(t, str) and t.strip() in exact_allocs):
            exact_hits += 1

    return {
        "n_total": n_total,
        "n_with_target": n_with_target,
        "n_null_target": n_total - n_with_target,
        "pool_hits": pool_hits,
        "pool_recall_target": (pool_hits / n_with_target) if n_with_target else 0.0,
        "pool_recall_total": (pool_hits / n_total) if n_total else 0.0,
        "window_hits": window_hits,
        "window_recall_target": (window_hits / n_with_target) if n_with_target else 0.0,
        "exact_hits": exact_hits,
        "exact_recall_target": (exact_hits / n_with_target) if n_with_target else 0.0,
    }
