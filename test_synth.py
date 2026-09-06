"""
Tests for the synthetic dataset harness and its oracle.

The load-bearing property is that **the model cannot influence the answer key**.
Most of these exist to pin that boundary: texture in, texture only; and every
arithmetic claim re-derived before a file is allowed to exist.

Runs under pytest, or standalone (`python test_synth.py`).
"""

import json
import tempfile
from pathlib import Path

import adapters
import synth
from synth_schema import EXACT_KINDS, summarise, validate


def small(seed=1, n=60, parties=6):
    return synth.generate(seed=seed, n_payments=n, n_parties=parties)


# --- the oracle --------------------------------------------------------------

def test_a_generated_dataset_validates():
    assert validate(small()) == []


def test_the_oracle_catches_a_tampered_amount():
    """The point of validating rather than trusting: change one invoice and the
    truth row that depends on it must stop being accepted."""
    dataset = small()
    paid = next(r for r in dataset["truth"] if r["invoice_ids"])
    target = paid["invoice_ids"][0]
    for invoice in dataset["invoices"]:
        if invoice["id"] == target:
            invoice["amount_cents"] += 1
    problems = validate(dataset)
    assert problems and "residual" in problems[0]


def test_the_oracle_catches_double_spending_an_invoice():
    """An invoice settled by two payments is cash applied twice."""
    dataset = small()
    rows = [r for r in dataset["truth"] if r["invoice_ids"]]
    rows[1]["invoice_ids"] = list(rows[0]["invoice_ids"])
    assert any("claimed by 2 payments" in p for p in validate(dataset))


def test_the_oracle_catches_a_dangling_reference():
    dataset = small()
    next(r for r in dataset["truth"] if r["invoice_ids"])["invoice_ids"] = ["NOPE"]
    assert any("unknown invoice" in p for p in validate(dataset))


def test_exact_kinds_really_tie():
    dataset = small()
    amounts = {i["id"]: i["amount_cents"] for i in dataset["invoices"]}
    payments = {p["id"]: p["amount_cents"] for p in dataset["payments"]}
    for row in dataset["truth"]:
        if row["kind"] in EXACT_KINDS:
            assert sum(amounts[i] for i in row["invoice_ids"]) == payments[row["payment_id"]]


def test_a_dataset_that_fails_validation_is_never_written():
    dataset = small()
    dataset["payments"][0]["amount_cents"] += 999
    with tempfile.TemporaryDirectory() as tmp:
        try:
            synth.write(dataset, Path(tmp) / "bad.json")
        except ValueError:
            assert not list(Path(tmp).glob("*.json")), "invalid dataset hit disk"
            return
    raise AssertionError("invalid dataset was accepted")


# --- the model cannot reach the answer key -----------------------------------

def test_texture_sanitiser_discards_numbers_the_model_volunteers():
    """The boundary. Anything beyond the requested strings is dropped, so a
    chatty reply degrades prose and nothing else."""
    clean = synth._sanitise_texture([{
        "name": "Acme", "reference_style": "INV-{n}", "memo_template": "PAY {refs}",
        "remittance_habit": "always",
        "amount_cents": 999_999, "invoice_ids": ["INV1"], "truth": "nonsense",
    }])
    assert clean == [{"name": "Acme", "reference_style": "INV-{n}",
                      "memo_template": "PAY {refs}", "remittance_habit": "always"}]


def test_texture_sanitiser_skips_malformed_entries():
    """A half-understood reply is not worth repairing."""
    assert synth._sanitise_texture([
        {"name": "", "reference_style": "INV-{n}", "memo_template": "x"},
        {"name": "NoPlaceholder", "reference_style": "INV-42", "memo_template": "x"},
        "not a dict", None,
    ]) == []


def test_an_unknown_habit_becomes_sometimes_not_an_error():
    clean = synth._sanitise_texture([{"name": "A", "reference_style": "{n}",
                                      "memo_template": "m", "remittance_habit": "wat"}])
    assert clean[0]["remittance_habit"] == "sometimes"


def test_generation_without_a_model_still_produces_exact_truth():
    """Losing the model costs realism in the prose, never correctness."""
    dataset = small()
    assert dataset["meta"]["texture_source"] == "fallback"
    assert dataset["meta"]["model"] is None
    assert validate(dataset) == []


def test_same_seed_same_truth():
    """Reproducible from the seed alone — texture varies, truth cannot."""
    a, b = small(seed=5), small(seed=5)
    assert a["truth"] == b["truth"]
    assert a["invoices"] == b["invoices"]
    assert a["payments"] == b["payments"]


def test_different_seeds_differ():
    assert small(seed=5)["truth"] != small(seed=6)["truth"]


# --- the cases that make this dataset worth generating -----------------------

def test_every_payment_carries_a_remitter_account():
    """The field BenchRec anonymised away, and the reason this exists at all."""
    dataset = small()
    assert all(p["remitter_account"] for p in dataset["payments"])
    assert len({p["remitter_account"] for p in dataset["payments"]}) > 1


def test_decoy_cases_really_contain_a_rival_group():
    """A GROUP_WITH_DECOY must be genuinely ambiguous on arithmetic alone,
    otherwise it does not test what rival counting exists to test."""
    from assemble import Line, Payment, enumerate_solutions

    dataset = small(seed=3, n=120, parties=6)
    by_account = {c["id"]: c["account"] for c in dataset["counterparties"]}
    invoices = {i["id"]: i for i in dataset["invoices"]}
    payments = {p["id"]: p for p in dataset["payments"]}

    checked = 0
    for row in dataset["truth"]:
        if row["kind"] != "GROUP_WITH_DECOY":
            continue
        payment = payments[row["payment_id"]]
        pool = [Line.of(i["id"], i["allocation"], i["amount_cents"] / 100)
                for i in dataset["invoices"]
                if by_account[i["counterparty_id"]] == payment["remitter_account"]]
        if len(pool) > 40:
            continue
        solutions, _ = enumerate_solutions(
            pool, Payment.of(payment["id"], payment["amount_cents"] / 100))
        assert len(solutions) >= 2, f"{row['payment_id']} has no rival group"
        checked += 1
    assert checked, "no decoy cases were checkable"


def test_the_mix_contains_every_kind():
    """A small dataset must still exercise every case, or it silently stops
    testing whatever it happened to omit."""
    kinds = {row["kind"] for row in small(n=80)["truth"]}
    assert kinds == set(synth.DEFAULT_MIX)


def test_no_match_cases_claim_nothing():
    for row in small()["truth"]:
        if row["kind"] == "NO_MATCH":
            assert row["invoice_ids"] == []


def test_amounts_are_integer_cents_not_floats():
    dataset = small()
    assert all(isinstance(i["amount_cents"], int) for i in dataset["invoices"])
    assert all(isinstance(p["amount_cents"], int) for p in dataset["payments"])


# --- the adapter into the cascade -------------------------------------------

def test_adapter_blocks_by_remitter_and_keeps_pools_searchable():
    """The whole reason for the remitter key: pools small enough to search."""
    dataset = small(n=120, parties=8)
    items = adapters.to_items(dataset)
    candidates = adapters.candidates_for(dataset)
    sizes = [len(candidates(i)) for i in items]
    assert max(sizes) < 400, "blocking is not narrowing at all"
    assert sum(sizes) / len(sizes) < 200


def test_adapter_never_offers_an_invoice_from_another_payer():
    dataset = small(n=80, parties=8)
    account_of = {c["id"]: c["account"] for c in dataset["counterparties"]}
    allowed = {i["allocation"]: account_of[i["counterparty_id"]]
               for i in dataset["invoices"]}
    candidates = adapters.candidates_for(dataset)
    for item in adapters.to_items(dataset):
        for c in candidates(item):
            assert allowed[c.allocation] == item.meta["remitter_account"]


def test_adapter_truth_is_expressed_in_allocations():
    """Resolutions carry allocations, so truth must too, or scoring compares
    two different things and silently reports zero."""
    dataset = small()
    allocations = {i["allocation"] for i in dataset["invoices"]}
    for want in adapters.truth_map(dataset).values():
        assert set(want) <= allocations


def test_round_trip_through_disk():
    dataset = small()
    with tempfile.TemporaryDirectory() as tmp:
        path = synth.write(dataset, Path(tmp) / "d.json")
        assert adapters.load(path)["truth"] == dataset["truth"]


def test_loading_a_corrupted_file_is_refused():
    """Validated on load as well as write: a file can be edited in between."""
    dataset = small()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.json"
        dataset["payments"][0]["amount_cents"] += 5000
        path.write_text(json.dumps(dataset), encoding="utf-8")
        try:
            adapters.load(path)
        except ValueError:
            return
    raise AssertionError("a corrupted dataset loaded without complaint")


def test_summarise_reports_the_mix():
    text = summarise(small())
    assert "GROUP_WITH_DECOY" in text and "case kinds" in text


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:
            failed.append(name)
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
