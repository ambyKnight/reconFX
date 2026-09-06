"""
Regression tests for precision monotonicity as confidence threshold rises.

Per PIPELINE.md Part 5:
"Monotonicity: precision must not decrease as the confidence threshold rises.
Today it does — §1.1 is exactly this test failing, and nothing was checking."
"""

import pandas as pd
import pytest

from recon.blocking import index_ledger
from recon.config import CONF_COLLISION_DEMOTED, CONF_SINGLE_ALLOCATION
from recon.decide import choose, match


def compute_confidence_frontier(df):
    """Compute cumulative precision at each confidence cutoff in descending order."""
    cuts = sorted(df["confidence"].dropna().unique(), reverse=True)
    frontier = []
    for cut in cuts:
        subset = df[df["confidence"] >= cut]
        prec = subset["ok"].mean() if len(subset) else 1.0
        frontier.append((cut, len(subset), prec))
    return frontier


def assert_precision_monotonicity(frontier, tolerance=1e-5):
    """Assert that precision is non-decreasing as confidence threshold rises.

    frontier is list of (cutoff, count, precision) sorted by cutoff descending.
    Moving backwards from lowest cutoff to highest cutoff, precision should not decrease.
    """
    # From low threshold to high threshold, precision should increase (or remain equal)
    ascending_by_cutoff = list(reversed(frontier))
    for i in range(len(ascending_by_cutoff) - 1):
        low_cut, low_n, low_prec = ascending_by_cutoff[i]
        high_cut, high_n, high_prec = ascending_by_cutoff[i + 1]
        assert high_prec >= low_prec - tolerance, (
            f"Monotonicity violated: threshold {high_cut:.2f} has precision {high_prec:.4f}, "
            f"which is lower than threshold {low_cut:.2f} precision {low_prec:.4f}!"
        )


def test_monotonicity_on_predictions_csv():
    """Verify monotonicity on generated predictions.csv."""
    try:
        df = pd.read_csv("predictions.csv")
    except FileNotFoundError:
        pytest.skip("predictions.csv not generated yet")

    if "ok" not in df.columns or "truth" not in df.columns:
        from recon.ingest import load_eval
        _, _, truth, _, _ = load_eval()
        df["truth"] = df["B_id"].astype(str).map({str(k): v for k, v in truth.items()})
        df["ok"] = (
            (df["pred"].fillna("").str.strip() == df["truth"].fillna("").str.strip())
            & df["pred"].notna()
        )

    attempted = df[df["pred"].notna()].copy()
    frontier = compute_confidence_frontier(attempted)
    assert_precision_monotonicity(frontier)


def test_monotonicity_bug_reproduction():
    """Demonstrate that the original collision bug violates monotonicity, and the fix restores it."""
    # Synthetic batch of items:
    # - 100 high-confidence regular items: 100% correct, conf 0.95
    # - 100 high-collision batch items: 100% correct, collisions=15
    # - 50 medium-confidence items: 98% correct (49/50), conf 0.80
    # - 50 ambiguous items: 20% correct (10/50), conf 0.35

    # With original buggy rule:
    # The 100 high-collision items (100% correct) are demoted to conf 0.50.
    # Bucket 0.50 has precision 100.0%.
    # Bucket 0.80 has precision 98.0%.
    # Cutoff >= 0.80 has precision: (100 + 49) / 150 = 99.33%
    # Cutoff >= 0.50 has precision: (100 + 49 + 100) / 250 = 99.60%!
    # Notice: cutoff 0.50 has HIGHER precision than cutoff 0.80! Monotonicity is violated!

    buggy_records = (
        [{"confidence": 0.95, "ok": True} for _ in range(100)]
        + [{"confidence": 0.80, "ok": True} for _ in range(49)]
        + [{"confidence": 0.80, "ok": False} for _ in range(1)]
        + [{"confidence": 0.50, "ok": True} for _ in range(100)]  # demoted 100% correct batch
        + [{"confidence": 0.35, "ok": True} for _ in range(10)]
        + [{"confidence": 0.35, "ok": False} for _ in range(40)]
    )
    df_buggy = pd.DataFrame(buggy_records)
    frontier_buggy = compute_confidence_frontier(df_buggy)

    # The buggy frontier MUST fail monotonicity test
    with pytest.raises(AssertionError, match="Monotonicity violated"):
        assert_precision_monotonicity(frontier_buggy)

    # With fixed rule:
    # High-collision items (15 collisions) remain at conf 0.95.
    fixed_records = (
        [{"confidence": 0.95, "ok": True} for _ in range(200)]  # 100 regular + 100 batch
        + [{"confidence": 0.80, "ok": True} for _ in range(49)]
        + [{"confidence": 0.80, "ok": False} for _ in range(1)]
        + [{"confidence": 0.35, "ok": True} for _ in range(10)]
        + [{"confidence": 0.35, "ok": False} for _ in range(40)]
    )
    df_fixed = pd.DataFrame(fixed_records)
    frontier_fixed = compute_confidence_frontier(df_fixed)

    # The fixed frontier MUST satisfy monotonicity
    assert_precision_monotonicity(frontier_fixed)
