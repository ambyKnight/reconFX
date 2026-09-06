"""
Stage 8: Append-only audit journal.

Job: Immutable recording of every decision for full auditable traceability.
Per ARCHITECTURE.md sections 4 & 6.11 and PIPELINE.md stage 8:
- Append-only SQLite table: one row per decision
- Columns: run_id, B_id, verdict, chosen_allocation, confidence, reason, config_hash, timestamp, reversed_by, reverses_id
- Guarantee: Reversal is always a new row referencing the original, never an update/delete.
  Triggers at the SQLite engine level strictly forbid any UPDATE or DELETE operations.
- Every AUTO / REVIEW / SUSPENSE decision gets exactly one row.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union

from recon.config import ARTIFACTS_DIR, get_config_hash

DEFAULT_JOURNAL_DB = ARTIFACTS_DIR / "journal.db"


class Journal:
    """Append-only SQLite journal for reconciliation decisions and audit log."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        if db_path is None:
            self.db_path = DEFAULT_JOURNAL_DB
        elif str(db_path) == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)

        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Obtain a SQLite database connection with row factory configured."""
        if self._conn is None or self.db_path == ":memory:":
            if self._conn is None:
                conn = sqlite3.connect(
                    str(self.db_path),
                    detect_types=sqlite3.PARSE_DECLTYPES,
                    check_same_thread=False,
                )
                conn.row_factory = sqlite3.Row
                self._conn = conn
            return self._conn
        return self._conn

    def _init_db(self) -> None:
        """Create the journal table and append-only enforcement triggers."""
        conn = self._get_connection()
        with conn:
            # 1. Create table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    B_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    chosen_allocation TEXT,
                    confidence REAL,
                    reason TEXT,
                    config_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    reversed_by TEXT,
                    reverses_id INTEGER
                );
                """
            )

            # 2. Indexes for fast lookup
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_b_id ON journal(B_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_run_id ON journal(run_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_verdict ON journal(verdict);")

            # 3. Triggers to strictly forbid UPDATE and DELETE (Append-only guarantee)
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_journal_no_update
                BEFORE UPDATE ON journal
                BEGIN
                    SELECT RAISE(FAIL, 'Updates not allowed on append-only journal table');
                END;
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_journal_no_delete
                BEFORE DELETE ON journal
                BEGIN
                    SELECT RAISE(FAIL, 'Deletes not allowed on append-only journal table');
                END;
                """
            )

    @staticmethod
    def _serialize_allocation(allocation: Any) -> Optional[str]:
        """Convert allocation set, list, or string to canonical JSON / text format."""
        if allocation is None:
            return None
        if isinstance(allocation, (list, set, tuple)):
            clean_list = sorted([str(x) for x in allocation if x is not None])
            return json.dumps(clean_list)
        return str(allocation)

    def log_decision(
        self,
        run_id: str,
        b_id: str,
        verdict: str,
        chosen_allocation: Optional[Union[str, list, set, tuple]] = None,
        allocation: Optional[Union[str, list, set, tuple]] = None,
        confidence: float = 0.0,
        reason: str = "",
        config_hash: Optional[str] = None,
        reversed_by: Optional[str] = None,
        reverses_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Log a single reconciliation decision into the immutable journal.

        Returns:
            The inserted row ID.
        """
        alloc_to_use = chosen_allocation if chosen_allocation is not None else allocation
        serialized_alloc = self._serialize_allocation(alloc_to_use)
        cfg_hash = config_hash or get_config_hash()
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO journal (
                    run_id, B_id, verdict, chosen_allocation, confidence,
                    reason, config_hash, timestamp, reversed_by, reverses_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(run_id),
                    str(b_id),
                    str(verdict),
                    serialized_alloc,
                    float(confidence) if confidence is not None else None,
                    str(reason or ""),
                    str(cfg_hash),
                    ts,
                    str(reversed_by) if reversed_by is not None else None,
                    int(reverses_id) if reverses_id is not None else None,
                ),
            )
            return cursor.lastrowid

    def log_decisions(
        self,
        decisions: List[Dict[str, Any]],
        run_id: Optional[str] = None,
    ) -> List[int]:
        """Batch-insert multiple reconciliation decisions in a single atomic transaction.

        Every AUTO / REVIEW / SUSPENSE item gets exactly one row.
        """
        inserted_ids = []
        conn = self._get_connection()
        with conn:
            for d in decisions:
                rid = d.get("run_id") or run_id
                if not rid:
                    raise ValueError("run_id must be provided per decision or as default")
                b_id = d.get("B_id") or d.get("b_id")
                verdict = d.get("verdict")
                alloc = d.get("chosen_allocation") or d.get("allocation") or d.get("pred")
                conf = d.get("confidence", 0.0)
                reason = d.get("reason") or d.get("tier") or ""
                cfg_hash = d.get("config_hash") or get_config_hash()
                ts = d.get("timestamp") or datetime.now(timezone.utc).isoformat()
                rev_by = d.get("reversed_by")
                rev_id = d.get("reverses_id")

                serialized_alloc = self._serialize_allocation(alloc)
                cursor = conn.execute(
                    """
                    INSERT INTO journal (
                        run_id, B_id, verdict, chosen_allocation, confidence,
                        reason, config_hash, timestamp, reversed_by, reverses_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        str(rid),
                        str(b_id),
                        str(verdict),
                        serialized_alloc,
                        float(conf) if conf is not None else None,
                        str(reason),
                        str(cfg_hash),
                        ts,
                        str(rev_by) if rev_by is not None else None,
                        int(rev_id) if rev_id is not None else None,
                    ),
                )
                inserted_ids.append(cursor.lastrowid)
        return inserted_ids

    def reverse_decision(
        self,
        original_id: int,
        run_id: str,
        reversed_by: str,
        reason: Optional[str] = None,
        config_hash: Optional[str] = None,
    ) -> int:
        """Reverse a prior decision by creating a NEW row referencing the original.

        Guaranteed: The original row is NEVER updated or deleted.
        """
        original = self.get_entry(original_id)
        if not original:
            raise ValueError(f"Decision with id {original_id} does not exist in journal")

        rev_reason = reason or f"Reversal of decision #{original_id} ({original['verdict']})"
        cfg_hash = config_hash or get_config_hash()
        ts = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO journal (
                    run_id, B_id, verdict, chosen_allocation, confidence,
                    reason, config_hash, timestamp, reversed_by, reverses_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(run_id),
                    original["B_id"],
                    "REVERSAL",
                    None,
                    original["confidence"],
                    rev_reason,
                    str(cfg_hash),
                    ts,
                    str(reversed_by),
                    int(original_id),
                ),
            )
            return cursor.lastrowid

    def get_open_items(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Identify items that remain in open suspense / review.

        An item is considered open if its latest event in the journal has verdict
        in ('SUSPENSE', 'REVIEW', 'NO-MATCH'), or if an AUTO item was subsequently reversed
        without being resolved again. Items that have been resolved via 'AUTO' or 'AUTO-CLEAR'
        are closed.

        Parameters:
            run_id: If provided, restricts to items whose suspense originated in this run
                    and remain unresolved.
        """
        conn = self._get_connection()
        # Query the latest event for each B_id
        if run_id:
            query = """
                WITH ranked AS (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY B_id ORDER BY id DESC) as rn
                    FROM journal
                ),
                run_bids AS (
                    SELECT DISTINCT B_id FROM journal
                    WHERE run_id = ? AND verdict IN ('SUSPENSE', 'REVIEW', 'NO-MATCH')
                )
                SELECT r.id, r.run_id, r.B_id, r.verdict, r.chosen_allocation,
                       r.confidence, r.reason, r.config_hash, r.timestamp,
                       r.reversed_by, r.reverses_id
                FROM ranked r
                JOIN run_bids rb ON r.B_id = rb.B_id
                WHERE r.rn = 1 AND r.verdict IN ('SUSPENSE', 'REVIEW', 'NO-MATCH', 'REVERSAL')
                ORDER BY r.id ASC;
            """
            rows = conn.execute(query, (str(run_id),)).fetchall()
        else:
            query = """
                WITH ranked AS (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY B_id ORDER BY id DESC) as rn
                    FROM journal
                )
                SELECT id, run_id, B_id, verdict, chosen_allocation,
                       confidence, reason, config_hash, timestamp,
                       reversed_by, reverses_id
                FROM ranked
                WHERE rn = 1 AND verdict IN ('SUSPENSE', 'REVIEW', 'NO-MATCH', 'REVERSAL')
                ORDER BY id ASC;
            """
            rows = conn.execute(query).fetchall()

        return [dict(r) for r in rows]

    def get_entries(
        self,
        run_id: Optional[str] = None,
        b_id: Optional[str] = None,
        verdict: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve journal entries with optional filtering."""
        conn = self._get_connection()
        clauses = []
        params = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(str(run_id))
        if b_id:
            clauses.append("B_id = ?")
            params.append(str(b_id))
        if verdict:
            clauses.append("verdict = ?")
            params.append(str(verdict))

        sql = "SELECT * FROM journal"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id ASC"

        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_entry(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single journal entry by its primary key ID."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM journal WHERE id = ?", (int(entry_id),)).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        """Return total count of records in the journal."""
        conn = self._get_connection()
        return conn.execute("SELECT COUNT(*) FROM journal").fetchone()[0]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Default journal singleton instance
_default_journal: Optional[Journal] = None


def get_default_journal(db_path: Optional[Union[str, Path]] = None) -> Journal:
    """Get or initialize default journal instance."""
    global _default_journal
    if _default_journal is None or (db_path and _default_journal.db_path != db_path):
        _default_journal = Journal(db_path)
    return _default_journal


def log_decision(
    run_id: str,
    b_id: str,
    verdict: str,
    allocation: Optional[Union[str, list, set, tuple]] = None,
    confidence: float = 0.0,
    reason: str = "",
    config_hash: Optional[str] = None,
    reversed_by: Optional[str] = None,
    reverses_id: Optional[int] = None,
    journal: Optional[Journal] = None,
) -> Dict[str, Any]:
    """Module-level helper to log a decision and return audit record dict.

    Maintains backwards compatibility with earlier stubs while persisting to SQLite.
    """
    j = journal or get_default_journal()
    row_id = j.log_decision(
        run_id=run_id,
        b_id=b_id,
        verdict=verdict,
        chosen_allocation=allocation,
        confidence=confidence,
        reason=reason,
        config_hash=config_hash,
        reversed_by=reversed_by,
        reverses_id=reverses_id,
    )
    return {
        "id": row_id,
        "run_id": run_id,
        "B_id": b_id,
        "verdict": verdict,
        "allocation": allocation,
        "confidence": confidence,
        "reason": reason,
        "config_hash": config_hash or get_config_hash(),
        "reversed_by": reversed_by,
        "reverses_id": reverses_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def reverse_decision(
    original_id: int,
    run_id: str,
    reversed_by: str,
    reason: Optional[str] = None,
    journal: Optional[Journal] = None,
) -> int:
    """Module-level helper to reverse a decision."""
    j = journal or get_default_journal()
    return j.reverse_decision(original_id, run_id, reversed_by, reason)


def get_open_items(run_id: Optional[str] = None, journal: Optional[Journal] = None) -> List[Dict[str, Any]]:
    """Module-level helper to retrieve still-open items from suspense."""
    j = journal or get_default_journal()
    return j.get_open_items(run_id=run_id)
