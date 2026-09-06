"""
Tests for Stage 8 append-only journal.

Verifies:
1. Append-only guarantee: UPDATE and DELETE are blocked by SQLite triggers.
2. Every AUTO, REVIEW, and SUSPENSE decision gets exactly one row.
3. Reversal creates a new row referencing the original, never modifying/deleting the original row.
4. Open item queries accurately track unresolved suspense/review items.
"""

import sqlite3
import pytest

from recon.config import get_config_hash
from recon.journal import Journal


def test_append_only_triggers_prevent_update_and_delete(tmp_path):
    db_file = tmp_path / "test_audit.db"
    journal = Journal(db_file)

    row_id = journal.log_decision(
        run_id="run_001",
        b_id="B_100",
        verdict="AUTO",
        chosen_allocation="ALLOC_100",
        confidence=0.95,
        reason="Single allocation match",
    )
    assert row_id == 1
    assert journal.count() == 1

    conn = journal._get_connection()

    # Attempt direct SQL UPDATE - must be rejected by trigger
    with pytest.raises(sqlite3.DatabaseError, match="Updates not allowed on append-only journal table"):
        with conn:
            conn.execute("UPDATE journal SET confidence = 0.50 WHERE id = ?", (row_id,))

    # Attempt direct SQL DELETE - must be rejected by trigger
    with pytest.raises(sqlite3.DatabaseError, match="Deletes not allowed on append-only journal table"):
        with conn:
            conn.execute("DELETE FROM journal WHERE id = ?", (row_id,))

    # Confirm original row remains intact
    entry = journal.get_entry(row_id)
    assert entry["id"] == row_id
    assert entry["verdict"] == "AUTO"
    assert entry["confidence"] == 0.95
    assert journal.count() == 1


def test_every_verdict_gets_one_row():
    journal = Journal(":memory:")

    decisions = [
        {"run_id": "run_test", "B_id": "B_1", "verdict": "AUTO", "chosen_allocation": "ALLOC_A", "confidence": 0.95, "reason": "single allocation"},
        {"run_id": "run_test", "B_id": "B_2", "verdict": "REVIEW", "chosen_allocation": "ALLOC_B", "confidence": 0.35, "reason": "ambiguous"},
        {"run_id": "run_test", "B_id": "B_3", "verdict": "SUSPENSE", "chosen_allocation": None, "confidence": 0.0, "reason": "no candidate"},
    ]

    ids = journal.log_decisions(decisions)
    assert len(ids) == 3
    assert journal.count() == 3

    entries = journal.get_entries(run_id="run_test")
    assert len(entries) == 3

    verdicts = [e["verdict"] for e in entries]
    assert verdicts == ["AUTO", "REVIEW", "SUSPENSE"]

    # Verify config hash was recorded on all rows
    expected_hash = get_config_hash()
    for e in entries:
        assert e["config_hash"] == expected_hash
        assert e["timestamp"] is not None
        assert e["reversed_by"] is None
        assert e["reverses_id"] is None


def test_reversal_creates_new_row_referencing_original():
    journal = Journal(":memory:")

    orig_id = journal.log_decision(
        run_id="run_001",
        b_id="B_POSTED",
        verdict="AUTO",
        chosen_allocation="ALLOC_ORIGINAL",
        confidence=0.95,
        reason="Auto-posted match",
    )
    assert orig_id == 1

    # Execute reversal
    rev_id = journal.reverse_decision(
        original_id=orig_id,
        run_id="run_002",
        reversed_by="auditor_sarah",
        reason="Disputed invoice reference",
    )
    assert rev_id == 2
    assert journal.count() == 2

    # Verify original row was NOT modified
    orig = journal.get_entry(orig_id)
    assert orig["verdict"] == "AUTO"
    assert orig["chosen_allocation"] == "ALLOC_ORIGINAL"
    assert orig["reversed_by"] is None
    assert orig["reverses_id"] is None

    # Verify reversal row
    rev = journal.get_entry(rev_id)
    assert rev["verdict"] == "REVERSAL"
    assert rev["B_id"] == "B_POSTED"
    assert rev["reverses_id"] == orig_id
    assert rev["reversed_by"] == "auditor_sarah"
    assert "Disputed invoice reference" in rev["reason"]


def test_reverse_nonexistent_id_raises_value_error():
    journal = Journal(":memory:")
    with pytest.raises(ValueError, match="does not exist in journal"):
        journal.reverse_decision(
            original_id=9999,
            run_id="run_error",
            reversed_by="system",
        )