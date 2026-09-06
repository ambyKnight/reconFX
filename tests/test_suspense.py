"""
Tests for Suspense subsystem, case file generation, aging, materiality, and categorizer.

Verifies:
1. Rule-based categorizer correctly classifies all 7 categories:
   - data error
   - suspected duplicate
   - genuine ambiguity
   - fee/FX residual
   - timing difference
   - unidentified receipt
   - missing ledger entry
2. Aging buckets (0-7, 8-30, 31-60, 60+ days).
3. Materiality and configurable write-off threshold.
4. Structured case file completeness (tiers tried, best candidates, why rejected).
"""

from collections import namedtuple
from datetime import datetime
import pandas as pd
import pytest

from recon.config import WRITE_OFF_THRESHOLD
from recon.suspense import (
    AGE_BUCKETS,
    CATEGORIES,
    assess_materiality,
    build_suspense_register,
    categorize_suspense,
    create_case_file,
    get_age_bucket,
)

BankItem = namedtuple(
    "BankItem",
    ["B_id", "amt", "date", "tokens", "counterparty", "transactionReferences", "collisions"],
    defaults=["B_test", 100.0, pd.Timestamp("2023-01-10"), set(), "ACME", "REF123", 1],
)
LedgerRow = namedtuple(
    "LedgerRow",
    ["A_id", "A_allocation", "amt", "date", "tokens", "A_counterparty"],
    defaults=["L_test", "ALLOC_1", 100.0, pd.Timestamp("2023-01-10"), set(), "ACME"],
)


def test_categorize_data_error():
    # Missing / NaN amount
    b_nan_amt = BankItem(amt=None)
    cat, reason = categorize_suspense(b_nan_amt)
    assert cat == "data error"

    # Zero amount
    b_zero = BankItem(amt=0.0)
    cat, reason = categorize_suspense(b_zero)
    assert cat == "data error"

    # Missing / NaT date
    b_nat = BankItem(amt=100.0, date=None)
    cat, reason = categorize_suspense(b_nat)
    assert cat == "data error"


def test_categorize_suspected_duplicate():
    # Multiple identical bank transactions competing for 1 ledger entry
    b = BankItem(amt=500.0, date=pd.Timestamp("2023-01-10"), collisions=3)
    cand = [LedgerRow(A_allocation="ALLOC_SINGLE", amt=500.0)]
    cat, reason = categorize_suspense(b, candidates=cand, duplicate_count=3)
    assert cat == "suspected duplicate"
    assert "competing" in reason


def test_categorize_genuine_ambiguity():
    # Multiple distinct candidate allocations tied
    b = BankItem(amt=250.0, date=pd.Timestamp("2023-01-10"))
    candidates = [
        LedgerRow(A_id="L1", A_allocation="ALLOC_ALPHA", amt=250.0),
        LedgerRow(A_id="L2", A_allocation="ALLOC_BETA", amt=250.0),
    ]
    cat, reason = categorize_suspense(b, candidates=candidates, tier="ambiguous, 2 allocations")
    assert cat == "genuine ambiguity"
    assert "Multiple candidate allocations" in reason


def test_categorize_fee_fx_residual():
    # Bank item has no exact match, but ledger has nearby amount within fee tolerance
    b = BankItem(amt=985.0, tokens={"INV88213"}, counterparty="SUPPLIER_X")
    by_amount = {
        1000.0: [
            LedgerRow(amt=1000.0, tokens={"INV88213"}, A_counterparty="SUPPLIER_X")
        ]
    }
    cat, reason = categorize_suspense(b, candidates=[], by_amount=by_amount, age_days=20)
    assert cat == "fee/FX residual"
    assert "Near amount match" in reason or "residual" in reason


def test_categorize_timing_difference():
    # Recent transaction (age <= 7 days) without ledger entry yet
    b = BankItem(amt=-1200.0, date=pd.Timestamp("2023-01-28"))
    cat, reason = categorize_suspense(b, candidates=[], age_days=3)
    assert cat == "timing difference"
    assert "recent" in reason


def test_categorize_unidentified_receipt():
    # Positive inbound receipt, older than 7 days, no matching candidates
    b = BankItem(amt=750.0, date=pd.Timestamp("2023-01-01"), tokens=set(), counterparty=None)
    cat, reason = categorize_suspense(b, candidates=[], age_days=35)
    assert cat == "unidentified receipt"
    assert "Inbound receipt" in reason


def test_categorize_missing_ledger_entry():
    # Outbound disbursement, older than 7 days, no matching candidates
    b = BankItem(amt=-4500.0, date=pd.Timestamp("2023-01-01"))
    cat, reason = categorize_suspense(b, candidates=[], age_days=25)
    assert cat == "missing ledger entry"
    assert "disbursement" in reason or "missing" in reason


def test_aging_buckets():
    assert get_age_bucket(0) == "0-7"
    assert get_age_bucket(7) == "0-7"
    assert get_age_bucket(8) == "8-30"
    assert get_age_bucket(30) == "8-30"
    assert get_age_bucket(31) == "31-60"
    assert get_age_bucket(60) == "31-60"
    assert get_age_bucket(61) == "60+"
    assert get_age_bucket(120) == "60+"
    assert get_age_bucket(None) == "unknown"


def test_materiality_and_write_off_threshold():
    # Using default threshold 25.00
    mat, is_wo = assess_materiality(15.50, write_off_threshold=25.00)
    assert mat == "immaterial"
    assert is_wo is True

    mat_neg, is_wo_neg = assess_materiality(-10.00, write_off_threshold=25.00)
    assert mat_neg == "immaterial"
    assert is_wo_neg is True

    mat_large, is_wo_large = assess_materiality(25.01, write_off_threshold=25.00)
    assert mat_large == "material"
    assert is_wo_large is False

    # Custom threshold
    mat_custom, is_wo_custom = assess_materiality(15.50, write_off_threshold=10.00)
    assert mat_custom == "material"
    assert is_wo_custom is False


def test_create_case_file():
    b = BankItem(
        B_id="B_CASE_01",
        amt=12.50,
        date=pd.Timestamp("2023-01-10"),
        tokens={"PAYMENT"},
        counterparty="CLIENT_A",
        collisions=1,
    )
    as_of = pd.Timestamp("2023-01-25")  # 15 days older -> bucket 8-30

    cf = create_case_file(
        bank_item=b,
        candidates=[],
        as_of_date=as_of,
        write_off_threshold=25.00,
        tier="no candidate",
        confidence=0.0,
    )

    assert cf.b_id == "B_CASE_01"
    assert cf.amount == 12.50
    assert cf.age_days == 15
    assert cf.age_bucket == "8-30"
    assert cf.is_write_off_candidate is True
    assert cf.materiality == "immaterial"
    assert cf.verdict == "SUSPENSE"
    assert len(cf.tiers_tried) > 0
    assert "No candidate" in cf.why_rejected


def test_build_suspense_register():
    b1 = BankItem(B_id="B1", amt=10.0, date=pd.Timestamp("2023-01-20"))
    b2 = BankItem(B_id="B2", amt=1000.0, date=pd.Timestamp("2023-01-01"))
    as_of = pd.Timestamp("2023-01-22")

    cf1 = create_case_file(b1, as_of_date=as_of, write_off_threshold=25.00)
    cf2 = create_case_file(b2, as_of_date=as_of, write_off_threshold=25.00)

    reg = build_suspense_register([cf1, cf2])
    assert reg["total_count"] == 2
    assert reg["total_value"] == 1010.0
    assert reg["write_off_candidates"]["count"] == 1
    assert reg["write_off_candidates"]["value"] == 10.0
    assert len(reg["by_category"]) == len(CATEGORIES)
    assert len(reg["by_aging"]) == len(AGE_BUCKETS)