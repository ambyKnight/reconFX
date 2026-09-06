"""
Tests for the collision demotion and promotion rules.

Verifies that:
1. High-collision items (>= promote_min, e.g. 12) are promoted to confident tier.
2. Risky middle-band collisions (e.g. 5 <= k < 12) are demoted to 0.50.
3. Low collisions (< 5) remain confident at 0.95.
"""

from collections import defaultdict
import pandas as pd
import pytest

from recon.decide import match


def test_collision_bands():
    # Build synthetic ledger and bank
    ledger = pd.DataFrame([
        {"A_id": "L1", "A_allocation": "ALLOC_A", "amt": 100.0, "date": pd.Timestamp("2023-01-01"), "tokens": set()}
    ])
    by_amount = {100.0: [row for row in ledger.itertuples()]}

    # Bank items with varying collision counts
    # Low collision (k=1)
    # Middle collision (k=8)
    # High collision (k=15)
    bank_rows = [
        {"B_id": "B_low", "amt": 100.0, "date": pd.Timestamp("2023-01-01"), "collisions": 1, "tokens": set()},
        {"B_id": "B_mid", "amt": 100.0, "date": pd.Timestamp("2023-01-01"), "collisions": 8, "tokens": set()},
        {"B_id": "B_high", "amt": 100.0, "date": pd.Timestamp("2023-01-01"), "collisions": 15, "tokens": set()},
    ]
    bank = pd.DataFrame(bank_rows)

    res = match(bank, by_amount, demote_min=5, promote_min=12)
    res_dict = dict(zip(res.B_id, res.confidence))
    tiers = dict(zip(res.B_id, res.tier))

    # Low collision stays 0.95
    assert res_dict["B_low"] == 0.95
    assert tiers["B_low"] == "single allocation"

    # Middle collision is demoted to 0.50
    assert res_dict["B_mid"] == 0.50
    assert "unique but 8 bank items" in tiers["B_mid"]

    # High collision is promoted to 0.95
    assert res_dict["B_high"] == 0.95
    assert tiers["B_high"] == "single allocation"


def test_custom_thresholds():
    ledger = pd.DataFrame([
        {"A_id": "L1", "A_allocation": "ALLOC_A", "amt": 50.0, "date": pd.Timestamp("2023-01-01"), "tokens": set()}
    ])
    by_amount = {50.0: [row for row in ledger.itertuples()]}

    bank = pd.DataFrame([
        {"B_id": "B1", "amt": 50.0, "date": pd.Timestamp("2023-01-01"), "collisions": 10, "tokens": set()},
    ])

    # If promote_min is 12, k=10 is demoted
    res1 = match(bank, by_amount, demote_min=5, promote_min=12)
    assert res1.iloc[0]["confidence"] == 0.50

    # If promote_min is 10, k=10 is promoted
    res2 = match(bank, by_amount, demote_min=5, promote_min=10)
    assert res2.iloc[0]["confidence"] == 0.95
