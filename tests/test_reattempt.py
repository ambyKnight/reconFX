"""
Tests for Re-attempt semantics (ARCHITECTURE.md Section 4.4).

Verifies:
1. Identifying still-open suspense/review items from prior runs in the journal.
2. Re-attempting matching against open items using updated config or expanded window.
3. Newly-resolved items are journalled as distinct AUTO-CLEAR events,
   preserving the original decisions intact without modification or deletion.
4. Open item queries accurately reflect that resolved items are no longer open.
"""

from collections import defaultdict
import pandas as pd
import pytest

from recon.blocking import index_ledger
from recon.journal import Journal
from recon.suspense import reattempt_open_suspense


def test_reattempt_auto_clears_open_suspense_item():
    journal = Journal(":memory:")

    # Date gap is 4 days: 2023-01-05 vs 2023-01-01
    bank_df = pd.DataFrame([
        {
            "B_id": "B_ITEM_001",
            "amt": 4200.0,
            "date": pd.Timestamp("2023-01-05"),
            "tokens": set(),
            "counterparty": "CLIENT_XYZ",
            "collisions": 1,
        }
    ])

    ledger_df = pd.DataFrame([
        {
            "A_id": "A_ITEM_001",
            "A_allocation": "ALLOC_RESOLVED_001",
            "amt": 4200.0,
            "date": pd.Timestamp("2023-01-01"),
            "tokens": set(),
            "counterparty": "CLIENT_XYZ",
        }
    ])
    by_amount = index_ledger(ledger_df)

    # 1. First run: narrow date window (1 day).
    # Since gap is 4 days, item finds no candidate within 1 day.
    orig_id = journal.log_decision(
        run_id="run_narrow_v1",
        b_id="B_ITEM_001",
        verdict="SUSPENSE",
        chosen_allocation=None,
        confidence=0.0,
        reason="no candidate within 1 day window",
    )
    assert orig_id == 1
    assert journal.count() == 1

    # Verify item is recognized as open
    open_items_initial = journal.get_open_items()
    assert len(open_items_initial) == 1
    assert open_items_initial[0]["B_id"] == "B_ITEM_001"
    assert open_items_initial[0]["verdict"] == "SUSPENSE"

    # 2. Re-attempt run under updated config (5-day window, auto_threshold=0.85)
    auto_cleared = reattempt_open_suspense(
        journal=journal,
        bank=bank_df,
        by_amount=by_amount,
        run_id="run_widened_v2",
        date_window_days=5,
        auto_threshold=0.85,
    )

    # Item should now match ALLOC_RESOLVED_001
    assert len(auto_cleared) == 1
    cleared_item = auto_cleared[0]
    assert cleared_item["B_id"] == "B_ITEM_001"
    assert cleared_item["allocation"] == "ALLOC_RESOLVED_001"
    assert cleared_item["confidence"] >= 0.85

    # 3. Verify append-only guarantees:
    # Original row must be unchanged
    orig_entry = journal.get_entry(orig_id)
    assert orig_entry["id"] == 1
    assert orig_entry["run_id"] == "run_narrow_v1"
    assert orig_entry["verdict"] == "SUSPENSE"
    assert orig_entry["chosen_allocation"] is None

    # New distinct row created for the AUTO-CLEAR event
    assert journal.count() == 2
    new_entry = journal.get_entry(cleared_item["id"])
    assert new_entry["id"] == 2
    assert new_entry["run_id"] == "run_widened_v2"
    assert new_entry["verdict"] == "AUTO-CLEAR"
    assert new_entry["chosen_allocation"] == "ALLOC_RESOLVED_001"
    assert "Auto-cleared" in new_entry["reason"]

    # 4. Open items query now shows item is resolved
    open_items_after = journal.get_open_items()
    assert len(open_items_after) == 0


def test_reattempt_leaves_unresolvable_item_open():
    journal = Journal(":memory:")

    bank_df = pd.DataFrame([
        {
            "B_id": "B_UNRESOLVABLE",
            "amt": 9999.0,
            "date": pd.Timestamp("2023-01-05"),
            "tokens": set(),
            "counterparty": "UNKNOWN",
            "collisions": 1,
        }
    ])
    # Empty ledger
    by_amount = defaultdict(list)

    journal.log_decision(
        run_id="run_v1",
        b_id="B_UNRESOLVABLE",
        verdict="SUSPENSE",
        chosen_allocation=None,
        confidence=0.0,
        reason="no candidate",
    )

    auto_cleared = reattempt_open_suspense(
        journal=journal,
        bank=bank_df,
        by_amount=by_amount,
        run_id="run_v2",
        date_window_days=5,
    )

    assert len(auto_cleared) == 0
    assert journal.count() == 1
    open_items = journal.get_open_items()
    assert len(open_items) == 1
    assert open_items[0]["B_id"] == "B_UNRESOLVABLE"