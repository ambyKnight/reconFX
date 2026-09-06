"""
Stage 7: Explanations.

Job: Generate human-readable reason strings for AUTO postings and REVIEW queues
explaining the specific evidence behind each decision.
"""


def explain_decision(bank_item, candidate, tier, confidence):
    """Generate a template-driven explanation string for a decision."""
    if tier == "single allocation":
        return f"Exact amount tie on same date ({bank_item.amt} on {bank_item.date.strftime('%Y-%m-%d') if hasattr(bank_item.date, 'strftime') else bank_item.date}) collapsing to single allocation."
    elif "date window" in tier:
        return f"Exact amount tie within date window."
    elif "reference tokens" in tier:
        return f"Multiple candidates separated by shared reference tokens."
    elif "ambiguous" in tier:
        return f"Ambiguous candidates requiring reviewer adjudication."
    return tier
