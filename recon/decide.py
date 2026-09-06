"""
Stage 6: Decision logic, tier assignment, and collision policy.

Assigns confidence and tier to shortlisted candidates, applying date-window
penalties and the corrected collision-band promotion/demotion rule.
"""

from collections import defaultdict
import pandas as pd

from recon.blocking import collision_counts, get_candidate_pool
from recon.config import (
    COLLISION_DEMOTE_MIN,
    COLLISION_PROMOTE_MIN,
    CONF_AMBIGUOUS,
    CONF_COLLISION_DEMOTED,
    CONF_DATE_WINDOW_PENALTY,
    CONF_NO_CANDIDATE,
    CONF_REF_TOKEN_TIE_BREAK,
    CONF_SINGLE_ALLOCATION,
    DATE_WINDOW_DAYS,
    load_fitted_config,
)


def choose(candidates, b):
    """Return (allocation, confidence, tier) for one bank item.

    Collapses candidates by allocation to prevent spurious ambiguity when
    multiple ledger rows represent the same allocation.
    """
    if not candidates:
        return None, CONF_NO_CANDIDATE, "no candidate"

    # Collapse to distinct allocations
    groups = defaultdict(list)
    for a in candidates:
        groups[a.A_allocation].append(a)

    if len(groups) == 1:
        return next(iter(groups)), CONF_SINGLE_ALLOCATION, "single allocation"

    # Several allocations: see if reference tokens single one out
    b_tokens = getattr(b, "tokens", set()) or set()
    scored = sorted(
        (
            (len(b_tokens & set().union(*(getattr(a, "tokens", set()) for a in rows), set())), alloc)
            for alloc, rows in groups.items()
        ),
        reverse=True,
    )
    best, runner_up = scored[0], scored[1]
    if best[0] > 0 and best[0] > runner_up[0]:
        return best[1], CONF_REF_TOKEN_TIE_BREAK, "reference tokens break the tie"

    return scored[0][1], CONF_AMBIGUOUS, f"ambiguous, {len(groups)} allocations"


def match(
    bank,
    by_amount,
    demote_min=None,
    promote_min=None,
    date_window_days=None,
):
    """Match bank rows against indexed ledger using tiered rules and collision bands.

    Parameters:
      bank: Bank lines DataFrame.
      by_amount: Index mapping rounded amount -> ledger rows.
      demote_min: Minimum collisions to trigger demotion (defaults to config/fitted).
      promote_min: Minimum collisions to promote back to confident (defaults to config/fitted).
      date_window_days: Date search tolerance in days.
    """
    # Load fitted parameters if not explicitly provided
    fitted = load_fitted_config() or {}
    if demote_min is None:
        demote_min = fitted.get("collision_demote_min", COLLISION_DEMOTE_MIN)
    if promote_min is None:
        promote_min = fitted.get("collision_promote_min", COLLISION_PROMOTE_MIN)
    if date_window_days is None:
        date_window_days = fitted.get("date_window_days", DATE_WINDOW_DAYS)

    if "collisions" not in bank.columns:
        bank = bank.assign(collisions=collision_counts(bank))

    out = []
    for b in bank.itertuples():
        pool, pool_type = get_candidate_pool(b, by_amount, date_window_days=date_window_days)
        alloc, conf, tier = choose(pool, b)

        if pool_type == "near" and alloc is not None:
            conf -= CONF_DATE_WINDOW_PENALTY
            tier += " (date window)"

        # Collision rule:
        # Demote confident matches ONLY in the risky middle band [demote_min, promote_min).
        # High collision counts (>= promote_min) represent bulk sweeps / batch postings
        # and are retained at the confident tier.
        if conf >= CONF_SINGLE_ALLOCATION:
            if demote_min <= b.collisions < promote_min:
                conf = CONF_COLLISION_DEMOTED
                tier = f"unique but {b.collisions} bank items share amount+date"

        out.append({
            "B_id": b.B_id,
            "pred": alloc,
            "confidence": round(conf, 2),
            "tier": tier,
            "n_candidates": len(pool),
        })

    return pd.DataFrame(out)
