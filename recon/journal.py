"""
Stage 8: Append-only audit journal.

Job: Immutable recording of every decision for full auditable traceability.
"""

from datetime import datetime, timezone


def log_decision(run_id, b_id, verdict, allocation, confidence, reason, config_hash):
    """Placeholder for Stage 8 append-only journal logging."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "B_id": b_id,
        "verdict": verdict,
        "allocation": allocation,
        "confidence": confidence,
        "reason": reason,
        "config_hash": config_hash,
    }
