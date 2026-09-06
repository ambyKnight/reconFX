"""
Tests for the 1-6 cascade.

Weighted toward stage 4's gates, because that is where a confident sentence from
a model turns into a posted journal entry. The rest check that the cascade
degrades in the safe direction under every failure it can actually have: no API
key, a model that throws, a tier nobody has measured.

Runs under pytest, or standalone (`python test_pipeline.py`).
"""

from calibrate import Calibration
from pipeline import (AUTO, NO_MATCH, REVIEW, AdjudicatorStage, Candidate,
                      ExactMatchStage, GroupAssemblyStage, Item, Resolution,
                      TriageStage, accuracy_report, fit_from_run, run_cascade,
                      verify_arithmetic)


def item(id, amount, refs=""):
    return Item(id, amount, None, refs)


def cand(id, alloc, amount, refs=""):
    return Candidate(id, alloc, amount, None, refs)


def pool_of(*candidates):
    return lambda _item: list(candidates)


def cascade(items, candidates, stages, calibration=None):
    return run_cascade(items, candidates, stages, TriageStage(calibration))


# --- the cascade shape -------------------------------------------------------

def test_each_stage_only_sees_what_the_previous_could_not_settle():
    items = [item("exact", 10_000), item("group", 10_000)]
    pools = {
        "exact": [cand("l1", "A", 10_000)],
        "group": [cand("l2", "B", 6_000), cand("l3", "C", 4_000)],
    }
    run = cascade(items, lambda i: pools[i.id],
                  [ExactMatchStage(), GroupAssemblyStage()])
    assert run.consumed == {"exact": 1, "group": 1, "unresolved": 0}
    assert {r.item_id: r.stage for r in run.resolutions} == {
        "exact": "exact", "group": "group"}


def test_unresolved_items_are_no_match_not_silence():
    """"Nothing matches" is an answer the system must be able to give."""
    run = cascade([item("x", 999)], pool_of(cand("l1", "A", 5)),
                  [ExactMatchStage(), GroupAssemblyStage()])
    assert run.consumed["unresolved"] == 1
    assert [r.decision for r in run.resolutions] == [NO_MATCH]


def test_group_stage_refuses_to_break_a_tie_silently():
    """Two subsets tie the amount; stage 3 must pass, not pick."""
    pool = pool_of(cand("l1", "A", 4_000), cand("l2", "B", 6_000),
                   cand("l3", "C", 2_500), cand("l4", "D", 7_500))
    run = cascade([item("b1", 10_000)], pool, [GroupAssemblyStage()])
    assert run.consumed["group"] == 0
    assert run.resolutions[0].decision == NO_MATCH


def test_several_rows_of_one_allocation_are_not_ambiguity():
    pool = pool_of(cand("l1", "A", 10_000), cand("l2", "A", 10_000))
    run = cascade([item("b1", 10_000)], pool, [ExactMatchStage()])
    assert run.consumed["exact"] == 1
    assert run.resolutions[0].allocations == ("A",)


# --- stage 4, gate 1: the model's arithmetic is re-checked -------------------

def test_verify_arithmetic_accepts_a_claim_that_ties():
    pool = [cand("l1", "A", 6_000), cand("l2", "B", 4_000)]
    assert verify_arithmetic(("A", "B"), item("b1", 10_000), pool, 0)


def test_verify_arithmetic_rejects_a_claim_that_does_not_tie():
    pool = [cand("l1", "A", 6_000), cand("l2", "B", 3_000)]
    assert not verify_arithmetic(("A", "B"), item("b1", 10_000), pool, 0)


def test_a_fluent_but_wrong_model_verdict_is_refused():
    """The failure that matters: a confident, well-argued answer whose numbers
    do not add up. It must not become a posted match."""
    pool = pool_of(cand("l1", "A", 6_000), cand("l2", "B", 3_000))

    def model(_item, _candidates):
        return {"verdict": "MATCH_GROUP", "candidate_indices": [0, 1],
                "confidence": 0.99,
                "reason": "These two invoices clearly settle the payment.",
                "key_facts": ["amounts align"]}

    run = cascade([item("b1", 10_000)], pool, [AdjudicatorStage(model)])
    assert run.consumed["ai"] == 0
    assert run.resolutions[0].decision == NO_MATCH


def test_a_correct_model_verdict_is_accepted():
    pool = pool_of(cand("l1", "A", 6_000), cand("l2", "B", 4_000))

    def model(_item, _candidates):
        return {"verdict": "MATCH_GROUP", "candidate_indices": [0, 1],
                "confidence": 0.4, "reason": "Both invoices are for this payer.",
                "key_facts": []}

    run = cascade([item("b1", 10_000)], pool, [AdjudicatorStage(model)])
    assert run.consumed["ai"] == 1
    assert run.resolutions[0].allocations == ("A", "B")
    assert run.resolutions[0].features["arithmetic_verified"] is True


def test_model_abstentions_are_not_matches():
    for verdict in ("NO_MATCH", "UNSURE"):
        def model(_item, _candidates, v=verdict):
            return {"verdict": v, "candidate_indices": [], "confidence": 0.9,
                    "reason": "", "key_facts": []}
        run = cascade([item("b1", 10_000)],
                      pool_of(cand("l1", "A", 10_000)), [AdjudicatorStage(model)])
        assert run.consumed["ai"] == 0, verdict


def test_out_of_range_indices_do_not_crash_or_match():
    """A model naming candidate 47 of 2 is hallucinating; it must not take down
    the run, and it must not resolve anything."""
    def model(_item, _candidates):
        return {"verdict": "MATCH", "candidate_indices": [47], "confidence": 1.0,
                "reason": "", "key_facts": []}

    run = cascade([item("b1", 10_000)], pool_of(cand("l1", "A", 10_000)),
                  [AdjudicatorStage(model)])
    assert run.consumed["ai"] == 0


def test_a_model_that_throws_is_an_abstention():
    def model(_item, _candidates):
        raise TimeoutError("endpoint down")

    run = cascade([item("b1", 10_000)], pool_of(cand("l1", "A", 10_000)),
                  [AdjudicatorStage(model)])
    assert run.consumed["ai"] == 0
    assert run.resolutions[0].decision == NO_MATCH


def test_no_api_key_degrades_to_the_human_queue():
    """No model configured means more review, never more guessing."""
    run = cascade([item("b1", 10_000)], pool_of(cand("l1", "A", 9_000)),
                  [AdjudicatorStage(None)])
    assert run.consumed["ai"] == 0
    assert run.resolutions[0].decision == NO_MATCH


def test_the_facts_we_build_are_renderable_by_the_real_dossier():
    """Regression: the bridge to llm_adjudicator, which shipped broken.

    `build_facts` emitted cash-shaped keys; `build_dossier` hardcoded
    intercompany ones and raised KeyError on the first real call. Every other
    test passed regardless, because they all inject a stub model — so this is
    the one test that touches the real seam. Skipped if litellm/neatlogs are
    absent, since the model is an optional dependency.
    """
    try:
        from llm_adjudicator import build_dossier
    except ImportError:
        return  # optional dependency not installed

    from pipeline import build_facts
    it = item("p1", 420_000, "INV88213")
    facts = [build_facts(it, cand("l1", "A", 420_000, "INV88213"))]
    text = build_dossier({"amount": "4200.00", "reference": "INV88213"}, facts)
    assert "[0]" in text and "allocation=A" in text


def test_a_stage_that_is_down_is_reported_not_hidden():
    """A model erroring on everything must not read as a model abstaining.

    Both produce zero matches and a full review queue; only the warning tells
    them apart, and without it a broken deployment looks like a hard week.
    """
    def broken(_item, _candidates):
        raise RuntimeError("KeyError: 'entity'")

    run = cascade([item("b1", 10_000), item("b2", 20_000)],
                  pool_of(cand("l1", "A", 5)), [AdjudicatorStage(broken)])
    assert run.warnings, "a stage that errored on every item reported nothing"
    assert "STAGE IS DOWN" in run.warnings[0]
    assert "WARNING" in accuracy_report(run)


def test_an_occasional_error_is_reported_but_not_called_down():
    calls = {"n": 0}

    def flaky(_item, _candidates):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("slow")
        return {"verdict": "MATCH", "candidate_indices": [0], "confidence": 0.5,
                "reason": "", "key_facts": []}

    run = cascade([item("b1", 10_000), item("b2", 10_000)],
                  pool_of(cand("l1", "A", 10_000)), [AdjudicatorStage(flaky)])
    assert run.warnings and "STAGE IS DOWN" not in run.warnings[0]
    assert run.consumed["ai"] == 1


def test_a_healthy_run_reports_no_warnings():
    run = cascade([item("b1", 10_000)], pool_of(cand("l1", "A", 10_000)),
                  [ExactMatchStage()])
    assert run.warnings == []


# --- stage 5, gates 2 and 3: nobody scores themselves ------------------------

def test_the_models_self_reported_confidence_is_never_the_number():
    """It is recorded as a feature and ignored as a score — the 0.85 lesson."""
    pool = pool_of(cand("l1", "A", 10_000))

    def model(_item, _candidates):
        return {"verdict": "MATCH", "candidate_indices": [0], "confidence": 0.99,
                "reason": "certain", "key_facts": []}

    run = cascade([item("b1", 10_000)], pool, [AdjudicatorStage(model)])
    r = run.resolutions[0]
    assert r.features["model_confidence"] == 0.99
    assert r.confidence == 0.0          # never measured, so it earns nothing
    assert r.decision == REVIEW


def test_an_unmeasured_tier_goes_to_review_not_auto():
    run = cascade([item("b1", 10_000)], pool_of(cand("l1", "A", 10_000)),
                  [ExactMatchStage()], Calibration.fit([("other/tier", True)] * 5000))
    assert run.resolutions[0].decision == REVIEW


def test_a_measured_tier_can_reach_auto():
    """The gate is strict, not a veto — a tier with a real record does pass."""
    items = [item("b1", 10_000)]
    pool = pool_of(cand("l1", "A", 10_000))
    tier = cascade(items, pool, [ExactMatchStage()]).resolutions[0].tier

    calibration = Calibration.fit([(tier, True)] * 3000)
    run = cascade(items, pool, [ExactMatchStage()], calibration)
    assert run.resolutions[0].decision == AUTO


def test_stages_do_not_set_their_own_confidence():
    """Structural: only triage may assign a number."""
    resolved, _ = ExactMatchStage().run([item("b1", 10_000)],
                                        pool_of(cand("l1", "A", 10_000)))
    assert resolved[0].confidence is None
    assert resolved[0].decision is None


def test_tiers_separate_stages_from_each_other():
    """An AI match and an exact match must not share a track record."""
    pool = pool_of(cand("l1", "A", 10_000))

    def model(_item, _candidates):
        return {"verdict": "MATCH", "candidate_indices": [0], "confidence": 0.5,
                "reason": "", "key_facts": []}

    exact = cascade([item("b", 10_000)], pool, [ExactMatchStage()]).resolutions[0]
    ai = cascade([item("b", 10_000)], pool, [AdjudicatorStage(model)]).resolutions[0]
    assert exact.tier != ai.tier


# --- stage 6 -----------------------------------------------------------------

def test_accuracy_report_works_without_labels():
    run = cascade([item("b1", 10_000)], pool_of(cand("l1", "A", 10_000)),
                  [ExactMatchStage()])
    report = accuracy_report(run)
    assert "exact" in report and "items: 1" in report


def test_accuracy_report_scores_against_truth():
    items = [item("right", 10_000), item("wrong", 20_000)]
    pools = {"right": [cand("l1", "A", 10_000)], "wrong": [cand("l2", "B", 20_000)]}
    run = cascade(items, lambda i: pools[i.id], [ExactMatchStage()])
    report = accuracy_report(run, {"right": ("A",), "wrong": ("Z",)})
    assert "50.0%" in report


def test_calibration_can_be_fitted_from_a_labelled_run():
    items = [item(f"b{i}", 10_000) for i in range(10)]
    run = cascade(items, pool_of(cand("l1", "A", 10_000)), [ExactMatchStage()])
    cal = fit_from_run(run, {f"b{i}": ("A",) for i in range(10)})
    assert cal.buckets[run.resolutions[0].tier].observed == 1.0


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
