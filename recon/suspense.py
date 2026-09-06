"""
Stage 4 & ARCHITECTURE.md Section 4: Suspense Subsystem & Exceptions Register.

Job: Provide structured case files for every unmatched and REVIEW item,
with documented rule-based categorization, materiality assessment, aging,
and re-attempt/auto-clear semantics.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from recon.config import (
    CONF_AUTO_THRESHOLD,
    DATE_WINDOW_DAYS,
    WRITE_OFF_THRESHOLD,
    get_config_hash,
)
from recon.journal import Journal, get_default_journal

# The 7 canonical suspense categories per ARCHITECTURE.md Section 4.2
CATEGORIES = [
    "timing difference",
    "missing ledger entry",
    "fee/FX residual",
    "genuine ambiguity",
    "unidentified receipt",
    "suspected duplicate",
    "data error",
]

# Standard accounting aging buckets per ARCHITECTURE.md Section 4.3
AGE_BUCKETS = ["0-7", "8-30", "31-60", "60+"]


def get_age_bucket(days: Optional[int]) -> str:
    """Classify age in days into canonical aging buckets: 0-7, 8-30, 31-60, 60+."""
    if days is None:
        return "unknown"
    if days <= 7:
        return "0-7"
    elif days <= 30:
        return "8-30"
    elif days <= 60:
        return "31-60"
    else:
        return "60+"


def assess_materiality(
    amount: Optional[float],
    write_off_threshold: float = WRITE_OFF_THRESHOLD,
) -> Tuple[str, bool]:
    """Determine whether an item is material or an immaterial write-off candidate.

    Per ARCHITECTURE.md Section 4.5: Below a configurable threshold, propose a write-off
    rather than an investigation.
    """
    if amount is None or pd.isna(amount):
        return "unknown", False
    is_candidate = abs(float(amount)) <= write_off_threshold
    materiality = "immaterial" if is_candidate else "material"
    return materiality, is_candidate


def categorize_suspense(
    bank_item: Any,
    candidates: Optional[List[Any]] = None,
    by_amount: Optional[Dict[Any, List[Any]]] = None,
    duplicate_count: int = 1,
    age_days: Optional[int] = None,
    tier: str = "",
    confidence: float = 0.0,
    residual_tolerance: float = 25.0,
) -> Tuple[str, str]:
    """Rule-based categorizer for suspense and review items per ARCHITECTURE.md Section 4.2.

    Evaluates evidence in documented priority order:
      1. data error: Malformed, non-numeric, or missing amount/date.
      2. suspected duplicate: Multiple identical bank lines competing for fewer ledger entries.
      3. genuine ambiguity: 2+ candidate allocations that cannot be separated by references or date.
      4. fee/FX residual: Ledger candidates exist with small amount residual (<= tolerance or 5%).
      5. timing difference: Recent transaction (age <= 7 days) likely awaiting next close cycle.
      6. unidentified receipt: Inbound credit (amt > 0) with no candidate and old (age > 7 days).
      7. missing ledger entry: Outbound payment or older unmatched line without corresponding GL entry.

    Returns:
      (category, explanation_string)
    """
    candidates = candidates or []
    amt = getattr(bank_item, "amt", None)
    date = getattr(bank_item, "date", None)

    # 1. Data error: missing, zero, or corrupt amount/date
    if amt is None or pd.isna(amt) or amt == 0 or date is None or pd.isna(date):
        return "data error", "Transaction has missing, zero, or malformed amount/date."

    # 2. Suspected duplicate: multiple identical bank transactions competing for one ledger row
    b_collisions = getattr(bank_item, "collisions", duplicate_count)
    if b_collisions > 1 and len(candidates) > 0:
        distinct_allocs = {getattr(c, "A_allocation", None) for c in candidates}
        if len(distinct_allocs) < b_collisions:
            return (
                "suspected duplicate",
                f"{b_collisions} bank lines share exact amount and date, competing for {len(distinct_allocs)} allocation(s).",
            )

    # 3. Genuine ambiguity: multiple distinct candidate allocations tied or unseparated
    distinct_allocs = {getattr(c, "A_allocation", None) for c in candidates}
    if len(distinct_allocs) >= 2 or "ambiguous" in tier.lower():
        return (
            "genuine ambiguity",
            f"Multiple candidate allocations ({len(distinct_allocs)}) tied; reference tokens and date insufficient to separate.",
        )

    # 4. Fee / FX residual: near-amount match in ledger within residual tolerance
    if len(candidates) == 0 and by_amount:
        target_amt = float(amt)
        for cand_amt, cand_rows in by_amount.items():
            diff = abs(float(cand_amt) - target_amt)
            if 0.01 <= diff <= residual_tolerance or (target_amt != 0 and diff / abs(target_amt) <= 0.05):
                b_tokens = getattr(bank_item, "tokens", set()) or set()
                b_cp = str(getattr(bank_item, "counterparty", "") or "").upper()
                for cr in cand_rows:
                    c_tokens = getattr(cr, "tokens", set()) or set()
                    c_cp = str(getattr(cr, "A_counterparty", "") or "").upper()
                    if (b_tokens and c_tokens and bool(b_tokens & c_tokens)) or (b_cp and c_cp and b_cp == c_cp):
                        return (
                            "fee/FX residual",
                            f"Near amount match ({cand_amt} vs bank {target_amt}, delta {diff:.2f}) with shared identity/tokens.",
                        )
                if diff <= 5.00:
                    return (
                        "fee/FX residual",
                        f"Ledger entry exists within residual tolerance (delta {diff:.2f}); likely bank fee or FX deduction.",
                    )

    # 5. Timing difference: recent item (age <= 7 days) likely awaiting next close
    if age_days is not None and age_days <= 7:
        return (
            "timing difference",
            f"Transaction is recent ({age_days} days old); expected to clear when ledger entry is posted next period.",
        )

    # 6. Unidentified receipt: inbound credit with no matching record and aged > 7 days
    if amt > 0 and len(candidates) == 0 and (age_days is None or age_days > 7):
        return (
            "unidentified receipt",
            f"Inbound receipt of {amt} received {age_days or 'unknown'} days ago with no matching GL entry or reference.",
        )

    # 7. Missing ledger entry: outbound payment or older unmatched line
    return (
        "missing ledger entry",
        f"Bank disbursement of {amt} with no corresponding general ledger transaction recorded.",
    )


@dataclass
class SuspenseCaseFile:
    """Structured case file for unmatched or REVIEW bank reconciliation items."""

    b_id: str
    amount: float
    date: str
    age_days: Optional[int]
    age_bucket: str
    materiality: str
    is_write_off_candidate: bool
    category: str
    category_reason: str
    tiers_tried: List[str]
    best_candidates: List[Dict[str, Any]]
    why_rejected: str
    verdict: str  # 'REVIEW' or 'SUSPENSE'
    tier: str
    confidence: float
    counterparty: Optional[str] = None
    references: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_case_file(
    bank_item: Any,
    candidates: Optional[List[Any]] = None,
    as_of_date: Optional[Union[datetime, pd.Timestamp]] = None,
    write_off_threshold: float = WRITE_OFF_THRESHOLD,
    by_amount: Optional[Dict[Any, List[Any]]] = None,
    duplicate_count: int = 1,
    tier: str = "",
    confidence: float = 0.0,
    pred_alloc: Optional[str] = None,
) -> SuspenseCaseFile:
    """Construct a full structured case file for an unmatched or review item."""
    candidates = candidates or []
    b_id = str(getattr(bank_item, "B_id", ""))
    amt = getattr(bank_item, "amt", None)
    date_val = getattr(bank_item, "date", None)
    amt_float = float(amt) if amt is not None and pd.notna(amt) else None

    date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") and pd.notna(date_val) else str(date_val or "")
    age_days = None
    if as_of_date is not None and date_val is not None and pd.notna(date_val):
        try:
            as_of_ts = pd.to_datetime(as_of_date)
            date_ts = pd.to_datetime(date_val)
            age_days = max(0, (as_of_ts - date_ts).days)
        except Exception:
            age_days = None

    age_bucket = get_age_bucket(age_days)
    materiality, is_write_off = assess_materiality(amt_float, write_off_threshold)

    category, cat_reason = categorize_suspense(
        bank_item=bank_item,
        candidates=candidates,
        by_amount=by_amount,
        duplicate_count=duplicate_count,
        age_days=age_days,
        tier=tier,
        confidence=confidence,
    )

    tiers_tried = [
        "exact_amount_same_day",
        "exact_amount_date_window",
        "reference_token_tie_break",
        "collision_middle_band_check",
    ]

    best_candidates = []
    for c in candidates[:5]:
        c_alloc = getattr(c, "A_allocation", None)
        c_amt = getattr(c, "amt", getattr(c, "A_amount", None))
        c_date = getattr(c, "date", getattr(c, "A_valueDate", None))
        c_date_str = c_date.strftime("%Y-%m-%d") if hasattr(c_date, "strftime") and pd.notna(c_date) else str(c_date)
        best_candidates.append({
            "allocation": c_alloc,
            "amt": float(c_amt) if c_amt is not None and pd.notna(c_amt) else None,
            "date": c_date_str,
            "tokens": list(getattr(c, "tokens", set()) or set()),
        })

    if len(candidates) == 0:
        why_rejected = f"No candidate ledger rows found with exact amount {amt_float} within date window (+/- {DATE_WINDOW_DAYS} days)."
    elif "collision" in tier.lower():
        why_rejected = f"Candidate found ({pred_alloc}) but demoted due to middle collision band risk ({tier})."
    elif "ambiguous" in tier.lower() or len(best_candidates) > 1:
        why_rejected = f"{len(best_candidates)} candidate allocations found; reference tokens did not distinguish a unique winner."
    else:
        why_rejected = f"Confidence {confidence:.2f} fell below auto-approval threshold ({CONF_AUTO_THRESHOLD:.2f})."

    verdict = "REVIEW" if len(candidates) > 0 else "SUSPENSE"

    cp = getattr(bank_item, "counterparty", getattr(bank_item, "B_counterparty", None))
    refs = getattr(bank_item, "transactionReferences", getattr(bank_item, "B_transactionReferences", None))

    return SuspenseCaseFile(
        b_id=b_id,
        amount=amt_float if amt_float is not None else 0.0,
        date=date_str,
        age_days=age_days,
        age_bucket=age_bucket,
        materiality=materiality,
        is_write_off_candidate=is_write_off,
        category=category,
        category_reason=cat_reason,
        tiers_tried=tiers_tried,
        best_candidates=best_candidates,
        why_rejected=why_rejected,
        verdict=verdict,
        tier=tier or ("no candidate" if len(candidates) == 0 else "review"),
        confidence=confidence,
        counterparty=str(cp) if cp is not None and pd.notna(cp) else None,
        references=str(refs) if refs is not None and pd.notna(refs) else None,
    )


def reattempt_open_suspense(
    journal: Journal,
    bank: pd.DataFrame,
    by_amount: Dict[Any, List[Any]],
    run_id: str,
    date_window_days: int = DATE_WINDOW_DAYS,
    demote_min: int = 5,
    promote_min: int = 12,
    auto_threshold: float = CONF_AUTO_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Re-attempt matching on still-open suspense items from prior runs.

    Per ARCHITECTURE.md Section 4.4:
    - Suspense is re-attempted, not archived.
    - Resolves open items using current configuration and passes.
    - Newly-resolved items are journalled as distinct AUTO-CLEAR events,
      NEVER modifying or deleting the original journal entries.
    """
    open_records = journal.get_open_items()
    if not open_records:
        return []

    open_bids = {str(r["B_id"]) for r in open_records}
    bank_open = bank[bank.B_id.astype(str).isin(open_bids)].copy()
    if bank_open.empty:
        return []

    from recon.decide import match

    pred_reattempt = match(
        bank_open,
        by_amount,
        demote_min=demote_min,
        promote_min=promote_min,
        date_window_days=date_window_days,
    )

    auto_cleared = []
    for row in pred_reattempt.itertuples():
        if row.confidence >= auto_threshold and row.pred is not None and pd.notna(row.pred):
            reason = f"Auto-cleared from suspense under updated config: {row.tier}"
            new_id = journal.log_decision(
                run_id=run_id,
                b_id=str(row.B_id),
                verdict="AUTO-CLEAR",
                chosen_allocation=str(row.pred),
                confidence=float(row.confidence),
                reason=reason,
                config_hash=get_config_hash(),
            )
            auto_cleared.append({
                "id": new_id,
                "B_id": str(row.B_id),
                "allocation": str(row.pred),
                "confidence": float(row.confidence),
                "reason": reason,
                "tier": row.tier,
            })

    return auto_cleared


def build_suspense_register(case_files: List[SuspenseCaseFile]) -> Dict[str, Any]:
    """Aggregate case files into a summary exceptions register."""
    total_count = len(case_files)
    total_value = sum(abs(cf.amount) for cf in case_files)

    cat_counts = {c: 0 for c in CATEGORIES}
    cat_values = {c: 0.0 for c in CATEGORIES}
    for cf in case_files:
        cat = cf.category if cf.category in cat_counts else "missing ledger entry"
        cat_counts[cat] += 1
        cat_values[cat] += abs(cf.amount)

    cat_summary = [
        {
            "category": c,
            "count": cat_counts[c],
            "value": round(cat_values[c], 2),
            "share_pct": round(100.0 * cat_counts[c] / total_count, 1) if total_count else 0.0,
            "value_share_pct": round(100.0 * cat_values[c] / total_value, 1) if total_value else 0.0,
        }
        for c in CATEGORIES
    ]
    cat_summary.sort(key=lambda x: x["count"], reverse=True)

    age_counts = {b: 0 for b in AGE_BUCKETS}
    age_values = {b: 0.0 for b in AGE_BUCKETS}
    for cf in case_files:
        bucket = cf.age_bucket if cf.age_bucket in age_counts else "unknown"
        if bucket in age_counts:
            age_counts[bucket] += 1
            age_values[bucket] += abs(cf.amount)

    age_summary = [
        {
            "bucket": b,
            "count": age_counts[b],
            "value": round(age_values[b], 2),
            "share_pct": round(100.0 * age_counts[b] / total_count, 1) if total_count else 0.0,
            "value_share_pct": round(100.0 * age_values[b] / total_value, 1) if total_value else 0.0,
        }
        for b in AGE_BUCKETS
    ]

    write_off_cfs = [cf for cf in case_files if cf.is_write_off_candidate]
    write_off_count = len(write_off_cfs)
    write_off_value = sum(abs(cf.amount) for cf in write_off_cfs)

    return {
        "total_count": total_count,
        "total_value": round(total_value, 2),
        "by_category": cat_summary,
        "by_aging": age_summary,
        "write_off_candidates": {
            "count": write_off_count,
            "value": round(write_off_value, 2),
            "share_pct": round(100.0 * write_off_count / total_count, 1) if total_count else 0.0,
            "value_share_pct": round(100.0 * write_off_value / total_value, 1) if total_value else 0.0,
        },
        "case_files": case_files,
    }