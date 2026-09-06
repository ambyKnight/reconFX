"""
Tests for Stage 1 candidate recall measurement instrumentation.
"""

import pandas as pd
import pytest

from recon.blocking import measure_candidate_recall, parse_target_allocations


def test_parse_target_allocations():
    # Single allocation
    assert parse_target_allocations("USD_2023-01-01_ACC#1") == ["USD_2023-01-01_ACC#1"]
    # List of allocations
    assert parse_target_allocations("[USD_ALLOC_1, USD_ALLOC_2]") == ["USD_ALLOC_1", "USD_ALLOC_2"]
    # Null or empty
    assert parse_target_allocations(None) == []
    assert parse_target_allocations("") == []
    assert parse_target_allocations("nan") == ["nan"]


def test_candidate_recall_measurement():
    # Synthetic ledger:
    # L1: exact match same day (amt=100, date=2023-01-01, ALLOC_1)
    # L2: near date (amt=200, date=2023-01-03, ALLOC_2)
    # L3: out of window date (amt=300, date=2023-01-20, ALLOC_3)
    # L4: member of group (amt=400, date=2023-01-01, ALLOC_G1)
    ledger = pd.DataFrame([
        {"A_id": "L1", "A_allocation": "ALLOC_1", "amt": 100.0, "date": pd.Timestamp("2023-01-01"), "tokens": set()},
        {"A_id": "L2", "A_allocation": "ALLOC_2", "amt": 200.0, "date": pd.Timestamp("2023-01-03"), "tokens": set()},
        {"A_id": "L3", "A_allocation": "ALLOC_3", "amt": 300.0, "date": pd.Timestamp("2023-01-20"), "tokens": set()},
        {"A_id": "L4", "A_allocation": "ALLOC_G1", "amt": 400.0, "date": pd.Timestamp("2023-01-01"), "tokens": set()},
    ])
    by_amount = {
        100.0: [row for row in ledger[ledger.amt == 100.0].itertuples()],
        200.0: [row for row in ledger[ledger.amt == 200.0].itertuples()],
        300.0: [row for row in ledger[ledger.amt == 300.0].itertuples()],
        400.0: [row for row in ledger[ledger.amt == 400.0].itertuples()],
    }

    # Bank items:
    # B1: same day exact -> in pool
    # B2: near date (2 days diff) -> in pool
    # B3: out of date window (19 days diff) -> NOT in pool, but in exact
    # B4: group target [ALLOC_G1, ALLOC_G2] -> ALLOC_G1 is in pool -> hit!
    # B5: no match at all (amt=999) -> NOT in pool
    # B6: null target -> excluded from evaluable target denominator
    bank = pd.DataFrame([
        {"B_id": "B1", "amt": 100.0, "date": pd.Timestamp("2023-01-01")},
        {"B_id": "B2", "amt": 200.0, "date": pd.Timestamp("2023-01-01")},
        {"B_id": "B3", "amt": 300.0, "date": pd.Timestamp("2023-01-01")},
        {"B_id": "B4", "amt": 400.0, "date": pd.Timestamp("2023-01-01")},
        {"B_id": "B5", "amt": 999.0, "date": pd.Timestamp("2023-01-01")},
        {"B_id": "B6", "amt": 100.0, "date": pd.Timestamp("2023-01-01")},
    ])
    truth = {
        "B1": "ALLOC_1",
        "B2": "ALLOC_2",
        "B3": "ALLOC_3",
        "B4": "[ALLOC_G1, ALLOC_G2]",
        "B5": "ALLOC_UNKNOWN",
        "B6": None,
    }

    stats = measure_candidate_recall(bank, by_amount, truth, date_window_days=5)

    assert stats["n_total"] == 6
    assert stats["n_with_target"] == 5
    assert stats["n_null_target"] == 1

    # B1, B2, B4 are in candidate pool (3 hits out of 5 items with target)
    assert stats["pool_hits"] == 3
    assert pytest.approx(stats["pool_recall_target"], 0.001) == 3 / 5  # 60%
    assert pytest.approx(stats["pool_recall_total"], 0.001) == 3 / 6   # 50%

    # In exact amount: B1, B2, B3, B4 are in exact (4 hits)
    assert stats["exact_hits"] == 4
    assert pytest.approx(stats["exact_recall_target"], 0.001) == 4 / 5  # 80%
