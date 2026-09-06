"""
Tests for group assembly and calibration.

PIPELINE.md §Part 5 asks specifically for "synthetic one-to-many cases with
known answers, plus an explicit case with two valid subsets asserting the
verdict is REVIEW, not a silent pick" — `test_two_valid_subsets_is_not_a_pick`
is that case. The rest exist because each one corresponds to a way the
uniqueness claim can be false while every individual number in it is correct.

Runs under pytest, or standalone (`python test_assemble.py`) so the invariants
are checkable without a test dependency installed.
"""

from datetime import date

from assemble import (MAX_SOLUTIONS, Assembly, Line, Payment, assemble,
                      enumerate_solutions, explain, to_cents)
from calibrate import (REQUIRED_PRECISION, Calibration, observations_needed,
                       tier_key, wilson_lower_bound)


def line(id, alloc, amount, refs="", d=None):
    return Line.of(id, alloc, amount, d, refs)


# --- the bug this was written for -------------------------------------------

FOUR_WAY = [
    line("l1", "ALLOC_A", 100.00),
    line("l2", "ALLOC_B", 60.00),
    line("l3", "ALLOC_C", 40.00),
    line("l4", "ALLOC_D", 70.00),
    line("l5", "ALLOC_E", 30.00),
    line("l6", "ALLOC_F", 50.00),
    line("l7", "ALLOC_G", 50.00),
]
# {A}, {B,C}, {D,E}, {F,G} all sum to 100.00.


def test_finds_all_four_rival_subsets():
    """The count is the product. If this is 2, everything downstream is a lie."""
    sols, exhaustive = enumerate_solutions(FOUR_WAY, Payment.of("b1", 100.00))
    assert exhaustive
    assert len(sols) == 4, [s.allocations for s in sols]


def test_early_exit_cannot_produce_uniqueness():
    """The original defect, reproduced and asserted against.

    `max_solutions=2` is the exact setting that shipped. The search still stops
    at two — that part is fine and bounded — but it must report AMBIGUOUS with
    an inexhaustive flag, never UNIQUE. This is the regression test for a 0.85
    confidence on a 35%-precision tier.
    """
    result = assemble(FOUR_WAY, Payment.of("b1", 100.00), max_solutions=2)
    assert result.verdict == "AMBIGUOUS"
    assert result.features["search_exhaustive"] is False
    assert result.features["solution_unique"] is False
    assert "at least 2" in explain(result)


def test_node_budget_also_degrades_downward():
    """Any budget, not just the solution cap, must fail toward ambiguity."""
    result = assemble(FOUR_WAY, Payment.of("b1", 100.00), node_budget=3)
    assert result.verdict != "UNIQUE"
    assert result.features["search_exhaustive"] is False


def test_invariant_is_enforced_not_just_documented():
    """A hand-forged overconfident result must be rejected at runtime."""
    forged = Assembly("UNIQUE", None, [],
                      {"search_exhaustive": False, "n_solutions": 1})
    try:
        forged.assert_never_false_unique()
    except AssertionError:
        return
    raise AssertionError("a truncated UNIQUE was allowed through")


# --- what uniqueness should mean --------------------------------------------

def test_genuinely_unique_group():
    pool = [line("l1", "INV_1", 1200.00), line("l2", "INV_2", 3000.00),
            line("l3", "INV_3", 17.50)]
    result = assemble(pool, Payment.of("b1", 4200.00))
    assert result.verdict == "UNIQUE"
    assert result.chosen.allocations == ("INV_1", "INV_2")
    assert result.features["n_solutions"] == 1
    assert "the only combination" in explain(result)


def test_rows_sharing_an_allocation_are_one_answer_not_many():
    """match.py's core insight, carried into groups.

    Three ledger rows of the same allocation, each tying the amount, are one
    answer. Counting them as three rivals would suppress a match that is unique
    in the only sense that matters.
    """
    pool = [line("l1", "ALLOC_X", 100.00), line("l2", "ALLOC_X", 100.00),
            line("l3", "ALLOC_X", 100.00)]
    result = assemble(pool, Payment.of("b1", 100.00))
    assert result.verdict == "UNIQUE"
    assert result.features["n_solutions"] == 1


def test_single_line_and_group_compete_in_one_universe():
    """A lone invoice and a trio summing to the same amount are rivals.

    Scoring groups separately from singles would hide precisely the collisions
    being counted, so size-1 subsets are enumerated alongside larger ones.
    """
    pool = [line("l1", "SOLO", 90.00), line("l2", "P1", 60.00),
            line("l3", "P2", 30.00)]
    result = assemble(pool, Payment.of("b1", 90.00))
    assert result.verdict == "AMBIGUOUS"
    assert result.features["n_solutions"] == 2


def test_two_valid_subsets_is_not_a_pick():
    """PIPELINE.md §Part 5, verbatim requirement."""
    pool = [line("l1", "A", 40.00), line("l2", "B", 60.00),
            line("l3", "C", 25.00), line("l4", "D", 75.00)]
    result = assemble(pool, Payment.of("b1", 100.00))
    assert result.verdict == "AMBIGUOUS"
    assert result.chosen is not None, "a shortlist head is still useful to a reviewer"
    assert "needs a human" in explain(result)


# --- the second, independent signal -----------------------------------------

def test_reference_tokens_break_an_arithmetic_tie():
    """Two subsets sum correctly; only one is corroborated by the memo."""
    pool = [line("l1", "A", 40.00, refs="INV88213"), line("l2", "B", 60.00, refs="INV88213"),
            line("l3", "C", 25.00), line("l4", "D", 75.00)]
    payment = Payment.of("b1", 100.00, references="PAYMENT REF INV88213")
    result = assemble(pool, payment)
    assert result.verdict == "CORROBORATED"
    assert result.chosen.allocations == ("A", "B")
    assert result.features["token_margin"] > 0


def test_shared_token_on_both_sides_is_not_a_tie_break():
    """Overlap that fails to discriminate must not be treated as evidence.

    The margin is the signal, not the raw count — two subsets both matching the
    memo say nothing about which is real.
    """
    pool = [line("l1", "A", 40.00, refs="ACME"), line("l2", "B", 60.00, refs="ACME"),
            line("l3", "C", 25.00, refs="ACME"), line("l4", "D", 75.00, refs="ACME")]
    result = assemble(pool, Payment.of("b1", 100.00, references="ACME"))
    assert result.verdict == "AMBIGUOUS"
    assert result.features["token_margin"] == 0


# --- search correctness ------------------------------------------------------

def test_credit_notes_do_not_break_pruning():
    """A negative line means "sum already exceeds target" is not a valid prune.

    If the bound were inadmissible, {A,B} would be lost and {C} would report as
    unique — a false uniqueness caused by an optimisation rather than a cap.
    """
    pool = [line("l1", "A", 150.00), line("l2", "B", -50.00), line("l3", "C", 100.00)]
    result = assemble(pool, Payment.of("b1", 100.00))
    assert result.verdict == "AMBIGUOUS"
    assert {s.allocations for s in result.solutions} == {("A", "B"), ("C",)}


def test_a_solution_extending_into_another_solution_is_found():
    """Recording a hit must not stop the branch: {A} and {A,B,C} both tie."""
    pool = [line("l1", "A", 100.00), line("l2", "B", 50.00), line("l3", "C", -50.00)]
    sols, exhaustive = enumerate_solutions(pool, Payment.of("b1", 100.00))
    assert exhaustive
    assert {s.allocations for s in sols} == {("A",), ("A", "B", "C")}


def test_tolerance_admits_a_bank_fee():
    pool = [line("l1", "A", 4199.65)]
    assert assemble(pool, Payment.of("b1", 4200.00)).verdict == "NONE"
    result = assemble(pool, Payment.of("b1", 4200.00), tolerance_cents=50)
    assert result.verdict == "UNIQUE"
    assert result.features["residual_cents"] == -35


def test_money_is_integer_cents():
    """0.1 + 0.2 in floats does not equal 0.3; subset-sum on floats is unstable."""
    assert to_cents("0.10") + to_cents("0.20") == to_cents("0.30")
    pool = [line("l1", "A", 0.10), line("l2", "B", 0.20)]
    assert assemble(pool, Payment.of("b1", 0.30)).verdict == "UNIQUE"


def test_oversized_pool_abstains_rather_than_guessing():
    pool = [line(f"l{i}", f"A{i}", 10.00 + i) for i in range(60)]
    result = assemble(pool, Payment.of("b1", 100.00))
    assert result.verdict == "ABSTAIN"
    assert result.features["abstained_on_pool_size"] is True
    assert "not guessing" in explain(result)


def test_no_subset_ties_the_amount():
    pool = [line("l1", "A", 33.00), line("l2", "B", 41.00)]
    result = assemble(pool, Payment.of("b1", 100.00))
    assert result.verdict == "NONE"
    assert result.chosen is None


def test_determinism():
    """PIPELINE.md §Part 5: same input, same config ⇒ identical output."""
    p = Payment.of("b1", 100.00)
    a = assemble(FOUR_WAY, p)
    b = assemble(list(reversed(FOUR_WAY)), p)
    assert [s.allocations for s in a.solutions] == [s.allocations for s in b.solutions]


# --- calibration: the 0.85 could not happen here -----------------------------

def test_a_perfect_record_at_n148_still_cannot_auto_post():
    """The incident, in one assertion.

    The tier that shipped at 0.85 had n=148. Even with a flawless record that
    supports only ~0.975 — so the ceiling on any claim it could make was below
    the 99.8% bar before a single error was counted. It was never eligible.
    """
    bound = wilson_lower_bound(148, 148)
    assert 0.97 < bound < 0.98
    assert bound < REQUIRED_PRECISION


def test_the_actual_measured_tier_lands_in_review():
    """35% observed over 148 → REVIEW, and nowhere near 0.85."""
    records = [("group:unique/exact", i < 63) for i in range(148)]
    cal = Calibration.fit(records)
    assert cal.verdict("group:unique/exact") == "REVIEW"
    assert cal.confidence("group:unique/exact") < 0.50


def test_the_zero_precision_variant_is_not_averaged_away():
    """Main variant 42.6%, date-window variant 0%. Separate keys, separate fates.

    Pooled into one bucket they read as a mediocre 35% tier; kept apart, one of
    them is wrong every single time and can be retired on its own evidence.
    """
    records = ([("group:unique/exact", i < 40) for i in range(94)]
               + [("group:unique/date_window", False) for _ in range(54)])
    cal = Calibration.fit(records)
    assert cal.buckets["group:unique/date_window"].observed == 0.0
    assert cal.confidence("group:unique/date_window") == 0.0
    assert cal.verdict("group:unique/date_window") == "REVIEW"


def test_an_unmeasured_tier_earns_nothing():
    """Adding a tier cannot silently start auto-posting."""
    cal = Calibration.fit([("known", True)] * 10)
    assert cal.confidence("brand_new_tier") == 0.0
    assert cal.verdict("brand_new_tier") == "REVIEW"


def test_enough_clean_evidence_does_clear_the_bar():
    """The gate is strict, not impassable — otherwise it is just a veto."""
    need = observations_needed()
    assert 1500 < need < 2500, need
    cal = Calibration.fit([("solid", True)] * need)
    assert cal.verdict("solid") == "AUTO"
    assert Calibration.fit([("solid", True)] * (need - 1)).verdict("solid") == "REVIEW"


def test_truncated_searches_get_their_own_track_record():
    """A capped search must not borrow a completed search's history."""
    completed = assemble(FOUR_WAY[:1], Payment.of("b1", 100.00))
    capped = assemble(FOUR_WAY, Payment.of("b1", 100.00), max_solutions=2)
    assert tier_key(completed) != tier_key(capped)
    assert "truncated" in tier_key(capped)


def test_crossfit_demotes_a_small_clean_band_below_a_large_near_clean_one():
    """The BenchRec k=8-11 failure, in miniature.

    Pooled, a perfect 1,682-observation band earns 0.99772 and outranks a
    44,267-observation band at 99.81% (0.99768) — and on eval that small band
    scored 93.47%. Crediting the worst fold restores the sane ordering, because
    a band that is clean only in places cannot be clean in every fold.
    """
    small = [(i % 5, "small_clean", True) for i in range(1682)]
    large = [(i % 5, "large_solid", i % 533 != 0) for i in range(44267)]

    pooled = Calibration.fit([(k, ok) for _, k, ok in small + large])
    assert pooled.confidence("small_clean") > pooled.confidence("large_solid")

    crossfit = Calibration.fit_folds(small + large)
    assert crossfit.confidence("small_clean") < crossfit.confidence("large_solid")


def test_crossfit_leaves_a_uniformly_solid_band_alone():
    """The guard must cost little where behaviour really is uniform."""
    records = [(i % 5, "uniform", i % 1000 != 0) for i in range(20000)]
    pooled = Calibration.fit([(k, ok) for _, k, ok in records])
    crossfit = Calibration.fit_folds(records)
    assert pooled.confidence("uniform") - crossfit.confidence("uniform") < 0.01


def test_calibration_round_trips_for_the_run_manifest():
    cal = Calibration.fit([("a", True)] * 50 + [("a", False)] * 3 + [("b", True)] * 9)
    back = Calibration.from_json(cal.to_json())
    assert back.confidence("a") == cal.confidence("a")
    assert back.confidence("b") == cal.confidence("b")


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
