"""
The contract for a synthetic reconciliation dataset: JSON Schema plus an oracle.

Why this file exists separately from the generator
--------------------------------------------------
A dataset is only useful if you can say precisely what it promises. Keeping the
schema and its validator apart from the code that produces them means the
generator can be rewritten, or a dataset can arrive from somewhere else
entirely, and `validate()` still decides whether it is admissible. The
generator is one producer; this is the standard.

The division of labour, which is the whole design
-------------------------------------------------
**Python owns the truth. The model owns the texture.**

The model writes company names, reference formats and memo phrasing — the messy
human material it is genuinely good at inventing. It never decides an amount,
never decides which invoices a payment settles, and never writes a row of
`truth`. Those are constructed deterministically, so the answer key is exact by
construction rather than by assertion.

This is not fastidiousness. The literature on LLM-generated ground truth is
consistent that labels need a formal oracle for anything high-stakes, and the
failure mode is specific: if the model invents both the data and the answer key,
a matcher measured against it is being scored on the model's mistakes, and every
number you report is really a number about the model. `validate()` is that
oracle — it re-derives every arithmetic claim in the file and rejects the
dataset if any of them does not hold.

It also means a dataset can be regenerated from `meta.seed` alone: same seed,
same numbers, same truth. Only the texture varies with the model, and texture
cannot change an answer.

What the dataset deliberately contains
--------------------------------------
Real cash application is not a stream of exact ties, and a generator that only
emits clean matches produces a matcher that only works on clean data. The case
kinds below are drawn from what actually goes wrong in cash application:
payments arriving with no remittance information at all (the single most cited
pain point), short payments and deductions, unexpected credits, bank fees taken
in transit, and — the one this project cares about most — **decoy groups**: a
rival set of invoices that also sums to the payment. That case is what makes
`assemble.py`'s rival counting measurable, and it is exactly what BenchRec could
not be made to show.

And critically, every payment carries a `remitter_account`. AGENTS.md §3b
records that BenchRec anonymised that field to a single constant, which is why
group assembly could not be measured there at all. Its presence here is the
point of generating a dataset in the first place, not an incidental field.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

SCHEMA_VERSION = "1.0"

# Case kinds. Each names a way a payment relates to invoices; the generator
# emits a mix, and a matcher's per-kind accuracy is far more informative than
# one pooled number — the same argument as calibrate.py's fine-grained tiers.
KINDS = (
    "EXACT_SINGLE",      # one payment, one invoice, exact
    "GROUP",             # one payment settles several invoices
    "GROUP_WITH_DECOY",  # ...and a rival set also sums to the amount
    "FEE_DEDUCTED",      # bank took a fee in transit; payment is short by a little
    "SHORT_PAY",         # customer deducted a credit/dispute
    "OVERPAYMENT",       # customer paid more than invoiced
    "PARTIAL",           # part payment against one invoice
    "NO_REMITTANCE",     # exact group, but memo carries nothing usable
    "NO_MATCH",          # unapplied cash: nothing in the ledger explains it
)

# Kinds whose payment amount must equal the sum of its invoices exactly. The
# rest carry a residual, and `validate` checks that the residual is exactly the
# stated difference — so "approximately right" is never accepted.
EXACT_KINDS = frozenset({"EXACT_SINGLE", "GROUP", "GROUP_WITH_DECOY", "NO_REMITTANCE"})

DATASET_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "reconFX synthetic reconciliation dataset",
    "type": "object",
    "required": ["meta", "counterparties", "invoices", "payments", "truth"],
    "additionalProperties": False,
    "properties": {
        "meta": {
            "type": "object",
            "required": ["schema_version", "seed", "currency", "generated_at",
                         "texture_source"],
            "additionalProperties": True,
            "properties": {
                "schema_version": {"type": "string"},
                "seed": {"type": "integer"},
                "currency": {"type": "string", "minLength": 3, "maxLength": 3},
                "generated_at": {"type": "string"},
                # How much of the prose the model actually wrote. "mixed" means
                # a partial reply was topped up from fixtures — not a failure,
                # but a dataset must not claim more provenance than it has.
                "texture_source": {"type": "string",
                                   "enum": ["llm", "mixed", "fallback"]},
                "model": {"type": ["string", "null"]},
            },
        },
        "counterparties": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "name", "account"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string", "minLength": 1},
                    # The field BenchRec lacks. See module docstring.
                    "account": {"type": "string", "minLength": 1},
                    "reference_style": {"type": "string"},
                    "remittance_habit": {
                        "type": "string",
                        "enum": ["always", "sometimes", "never"]},
                    # Texture, kept in the dataset so a memo can be regenerated
                    # or audited without re-running the model.
                    "memo_template": {"type": "string"},
                },
            },
        },
        "invoices": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "counterparty_id", "allocation", "amount_cents",
                             "currency", "issue_date"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "counterparty_id": {"type": "string"},
                    "allocation": {"type": "string"},
                    # Integer cents everywhere. PIPELINE.md §Stage 0: money in
                    # float invites 1-cent tie failures, and a dataset that
                    # stores floats bakes that bug into every consumer.
                    "amount_cents": {"type": "integer"},
                    "currency": {"type": "string"},
                    "issue_date": {"type": "string"},
                    "due_date": {"type": ["string", "null"]},
                    "reference": {"type": "string"},
                },
            },
        },
        "payments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "counterparty_id", "amount_cents", "currency",
                             "value_date", "remitter_account"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "counterparty_id": {"type": "string"},
                    "amount_cents": {"type": "integer"},
                    "currency": {"type": "string"},
                    "value_date": {"type": "string"},
                    "remitter_account": {"type": "string"},
                    "remitter_name": {"type": "string"},
                    "memo": {"type": "string"},
                },
            },
        },
        "truth": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["payment_id", "invoice_ids", "kind", "residual_cents"],
                "additionalProperties": False,
                "properties": {
                    "payment_id": {"type": "string"},
                    "invoice_ids": {"type": "array", "items": {"type": "string"}},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    # payment - sum(invoices). Zero for EXACT_KINDS; otherwise the
                    # exact, intended difference — never a tolerance.
                    "residual_cents": {"type": "integer"},
                    "note": {"type": "string"},
                },
            },
        },
    },
}


def _check_unique_ids(dataset: dict, problems: list[str]) -> None:
    for section in ("counterparties", "invoices", "payments"):
        counts = Counter(row["id"] for row in dataset.get(section, []))
        for id_, n in counts.items():
            if n > 1:
                problems.append(f"{section}: id {id_!r} appears {n} times")


def _check_references(dataset: dict, problems: list[str]) -> None:
    """Referential integrity: nothing may point at something that isn't there."""
    parties = {c["id"] for c in dataset["counterparties"]}
    invoices = {i["id"] for i in dataset["invoices"]}
    payments = {p["id"] for p in dataset["payments"]}

    for invoice in dataset["invoices"]:
        if invoice["counterparty_id"] not in parties:
            problems.append(f"invoice {invoice['id']}: unknown counterparty")
    for payment in dataset["payments"]:
        if payment["counterparty_id"] not in parties:
            problems.append(f"payment {payment['id']}: unknown counterparty")
    for row in dataset["truth"]:
        if row["payment_id"] not in payments:
            problems.append(f"truth: unknown payment {row['payment_id']}")
        for invoice_id in row["invoice_ids"]:
            if invoice_id not in invoices:
                problems.append(
                    f"truth {row['payment_id']}: unknown invoice {invoice_id}")


def _check_arithmetic(dataset: dict, problems: list[str]) -> None:
    """The oracle. Re-derive every claim rather than trusting the generator.

    This is the guarantee that makes the dataset usable as an answer key: a
    `truth` row asserting three invoices settle a payment is only accepted if
    those three invoices actually sum to that payment, to the cent. A generator
    bug, or a model that slipped an amount past the boundary, fails here rather
    than quietly becoming a wrong label that some future matcher gets blamed for.
    """
    amounts = {i["id"]: i["amount_cents"] for i in dataset["invoices"]}
    payments = {p["id"]: p for p in dataset["payments"]}

    for row in dataset["truth"]:
        payment = payments.get(row["payment_id"])
        if payment is None:
            continue
        if row["kind"] == "NO_MATCH":
            if row["invoice_ids"]:
                problems.append(f"truth {row['payment_id']}: NO_MATCH has invoices")
            continue
        if not row["invoice_ids"]:
            problems.append(f"truth {row['payment_id']}: {row['kind']} has no invoices")
            continue

        total = sum(amounts.get(i, 0) for i in row["invoice_ids"])
        residual = payment["amount_cents"] - total
        if residual != row["residual_cents"]:
            problems.append(
                f"truth {row['payment_id']}: residual_cents={row['residual_cents']} "
                f"but payment - invoices = {residual}")
        if row["kind"] in EXACT_KINDS and residual != 0:
            problems.append(
                f"truth {row['payment_id']}: {row['kind']} must tie exactly, "
                f"off by {residual}")


def _check_exclusivity(dataset: dict, problems: list[str]) -> None:
    """No invoice may be settled by two payments.

    Mutual exclusivity is item 1 of AGENTS.md §10 and a real accounting
    constraint: an invoice settled twice is cash applied twice. A dataset that
    violates it would make a correct matcher look wrong.
    """
    claims: dict[str, list[str]] = defaultdict(list)
    for row in dataset["truth"]:
        for invoice_id in row["invoice_ids"]:
            claims[invoice_id].append(row["payment_id"])
    for invoice_id, payment_ids in claims.items():
        if len(payment_ids) > 1:
            problems.append(
                f"invoice {invoice_id} claimed by {len(payment_ids)} payments: "
                f"{', '.join(payment_ids)}")


def _check_shape(dataset: dict, problems: list[str]) -> None:
    """Minimal structural check against DATASET_SCHEMA's required keys.

    Deliberately not a full JSON Schema implementation: `jsonschema` is not a
    dependency of this repo and the schema is published for consumers who want
    to validate externally. What matters here is that the required keys exist
    and the types are usable, so the arithmetic oracle can run at all.
    """
    for key in DATASET_SCHEMA["required"]:
        if key not in dataset:
            problems.append(f"missing top-level key {key!r}")
    if problems:
        return

    meta = dataset["meta"]
    for key in DATASET_SCHEMA["properties"]["meta"]["required"]:
        if key not in meta:
            problems.append(f"meta: missing {key!r}")

    for invoice in dataset["invoices"]:
        if not isinstance(invoice.get("amount_cents"), int):
            problems.append(f"invoice {invoice.get('id')}: amount_cents must be int cents")
    for payment in dataset["payments"]:
        if not isinstance(payment.get("amount_cents"), int):
            problems.append(f"payment {payment.get('id')}: amount_cents must be int cents")
        if not payment.get("remitter_account"):
            # The whole reason this dataset exists; an empty one is a silent
            # regression back to the BenchRec situation.
            problems.append(f"payment {payment.get('id')}: empty remitter_account")


def validate(dataset: dict) -> list[str]:
    """Return a list of problems. Empty means the dataset is admissible.

    Returns rather than raises so a caller can report every problem at once;
    a generator that emits fifty bad rows should not be debugged fifty runs in
    a row.
    """
    problems: list[str] = []
    _check_shape(dataset, problems)
    if problems:
        return problems

    _check_unique_ids(dataset, problems)
    _check_references(dataset, problems)
    if problems:
        return problems

    _check_arithmetic(dataset, problems)
    _check_exclusivity(dataset, problems)
    return problems


def summarise(dataset: dict) -> str:
    """One-screen description of what a dataset actually contains."""
    kinds = Counter(row["kind"] for row in dataset["truth"])
    sizes = Counter(len(row["invoice_ids"]) for row in dataset["truth"]
                    if row["kind"] != "NO_MATCH")
    lines = [
        f"schema {dataset['meta']['schema_version']}  seed {dataset['meta']['seed']}  "
        f"texture {dataset['meta']['texture_source']}",
        f"{len(dataset['counterparties'])} counterparties, "
        f"{len(dataset['invoices'])} invoices, {len(dataset['payments'])} payments",
        "",
        "case kinds:",
    ]
    total = sum(kinds.values()) or 1
    for kind, n in kinds.most_common():
        lines.append(f"  {kind:<18} {n:>5}  ({100 * n / total:.1f}%)")
    lines.append("")
    lines.append("group sizes: " + ", ".join(
        f"{size}:{n}" for size, n in sorted(sizes.items())))
    return "\n".join(lines)
