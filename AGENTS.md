# AGENTS.md — read this first

Central index for anyone (human or agent) picking up this repo. It tells you
what the project is, what's actually built vs. only designed, what's verified
vs. assumed, and where to find the rest. If you're a new agent session working
on this repo, start here before reading anything else.

---

## 1. What this project is

**reconFX** — a reconciliation system that sorts every entry into "I'm sure,
post it," "I'm not sure, ask a human," or "nothing matches, say so" — instead of
guessing and sounding equally confident either way.

**Current target: intercompany reconciliation** — getting two entities of the
same corporate group to agree on what they owe each other (FX translation,
transfer-pricing markups, netting settlements, mismatched close calendars).
This is a pivot from an earlier phase that targeted bank-vs-ledger cash
reconciliation. Both phases' work is kept; see §3.

Full pitch: [PROJECT.md](PROJECT.md).

## 2. The one thing to not get wrong: evidence vs. design

This repo mixes two kinds of content and they must not be confused:

- **Measured evidence** — real numbers, computed from a real dataset
  ([BenchRec](https://www.kaggle.com/datasets/benchmarkteam/benchrec-real-world-cash-reconciliation-dataset),
  bank-vs-ledger cash reconciliation). Lives in [PIPELINE.md](PIPELINE.md) and
  the `research/`-shaped scripts in §4.
- **Designed-but-unmeasured** — the intercompany architecture in
  [ARCHITECTURE.md](ARCHITECTURE.md). No public intercompany dataset exists (we
  looked — see PROJECT.md's evidence section), so nothing there has been
  measured on real intercompany data yet. Numbers cited in ARCHITECTURE.md are
  explicitly carried over from the BenchRec evidence base as sizing estimates,
  not intercompany measurements — it says so inline every time.

**Rule for anyone adding to this repo: never let a number from one bucket get
cited as if it belongs to the other.** If you measure something new, say what
you measured it on.

## 3. What's actually built right now

| File | What it is | Status |
|---|---|---|
| [match.py](match.py) | Deterministic tiered matcher (amount+date collapse, collision handling, reference-token tie-break) | Working, evaluated on BenchRec. Collision rule now **fitted on train** rather than hand-set — see §3a. |
| [assemble.py](assemble.py) | Stage 2 group assembly: bounded subset-sum that **counts** rival solutions instead of stopping at the first, plus reference-token corroboration as a second signal | Working, 26/26 unit tests. **Not yet exercised on real data** — see §3a. |
| [calibrate.py](calibrate.py) | Confidence derived from measured outcomes (Wilson lower bound, cross-fitted across folds) instead of hand-typed literals | Working, unit-tested, wired into `match.py` |
| [test_assemble.py](test_assemble.py) | Tests for both of the above, incl. regression tests for the false-uniqueness bug | 26/26 passing; runs standalone (`python test_assemble.py`) or under pytest |
| [benchrec_probe.py](benchrec_probe.py) | Dataset/baseline exploration | Working, evidence-gathering script |
| [rule_archaeology.py](rule_archaeology.py) | Reverse-engineers the bank's own production matching rules from train labels | Working, evidence-gathering script |
| [error_autopsy.py](error_autopsy.py) | Buckets every confident-tier miss into why it happened (label noise vs. real error) | Working, evidence-gathering script |
| [sanity.py](sanity.py), [truth_shape.py](truth_shape.py) | Data-integrity checks on the BenchRec solution file | Working, evidence-gathering scripts |
| [llm_adjudicator.py](llm_adjudicator.py) | The LLM adjudication step from ARCHITECTURE.md §3 — takes a dossier, returns a schema-enforced verdict (MATCH/MATCH_GROUP/NO_MATCH/UNSURE) | **Live-tested against the real model, see §5.** Everything else in the pass loop (blocking, group assembly, calibration, decision gates, journal, open-items subsystem) is designed in ARCHITECTURE.md but not yet built. |

Nothing in the intercompany pass loop exists yet except this one adjudication
step. `llm_adjudicator.py`'s placeholder dossier in `main()` is fabricated for
testing — there is no real intercompany data feeding it.

## 3a. Measured results, 2026-09-06 (BenchRec — see §2 before citing these)

**Collision rule, PIPELINE.md §1.1, now fitted instead of hand-set.** The band is
chosen on `train` and applied to `eval` once, so unlike §1.1's 81.1% this is not
leakage:

| policy | match rate | precision |
|---|---|---|
| original hand-set `k >= 5` demote | 62.7% | 99.821% |
| fitted band, pooled Wilson bound | **0.0%** | — |
| **fitted band, cross-fitted (shipped)** | **76.2%** | **99.89%** |

PIPELINE.md §1.1 predicted "expect the honest number to land below 81.1%". It
does: 76.2%.

**The middle row is the finding, not a failure.** With one pooled Wilson bound,
band `k=8-11` scored 100% on all 1,682 train observations and earned 0.99772 —
outranking `k=1` and its 44,267 observations at 0.99768. On eval that same band
scored **93.47%**. A perfect record on a small band beat a near-perfect record on
a huge one, put the worst band at the top of the frontier, and took the whole
match rate to zero. Sampling error was not what went wrong, so no amount of
evidence-weighting inside a single pooled estimate could have caught it.

Cross-fitting across 5 folds (`Calibration.fit_folds`, crediting each band its
**worst** fold) fixes the ordering: k=1 (0.9964) > k=12+ (0.9880) > k=8-11
(0.9877). This is what `match.py` now ships. It is also a concrete argument for
PIPELINE.md §Stage 3's claim that `k` belongs in a model as a non-monotone
feature rather than in a policy of its own.

**Group assembly is built and tested but NOT yet exercised on real data.**
`assemble.py` passes 26/26 unit tests, but run against BenchRec's 1,779 real
list-targets it returned `ABSTAIN` on 1,777 of them (99.9%): blocking on
account + currency + 5-day window yields candidate pools far larger than
`MAX_POOL = 40`, so the subset-sum never runs. **This is a Stage 1 blocking
problem, not a Stage 2 problem, and it is the next thing to fix** — until it is,
the 5.6% group-target pool in PIPELINE.md §1.3 stays forfeited and no claim
about group precision has any measurement behind it. Note that abstaining is the
designed behaviour here rather than a crash; it is simply abstaining on
everything.

## 4. Model stack — read before writing any LLM code

**We do not use Anthropic/Claude for the reconciliation LLM loop.** This was an
explicit decision. The loop runs on a **custom-hosted model via TensorMux**:

- Endpoint: `https://api.tensormux.com/v1` (OpenAI-compatible `/chat/completions`)
- Model: `glm-4-7-flash`
- Called through **LiteLLM**: `litellm.completion(model="openai/glm-4-7-flash", api_base=..., api_key=...)` — this is LiteLLM's standard convention for any OpenAI-compatible custom endpoint, not TensorMux-specific plumbing. Swapping the backend later is a config change.
- Tool/function calling **is confirmed to work** against this model — see §5.

If you add a second LLM call anywhere in this codebase, match this pattern.
Don't introduce a different provider or a raw `openai`/`anthropic` SDK call
without a reason.

## 5. What's been verified live (not just assumed)

Run on 2026-09-06 against the real TensorMux endpoint:

- **Tool calling works.** `glm-4-7-flash` correctly called `record_verdict`
  with valid, schema-conforming arguments — this was previously an open
  question, now closed.
- **The verdict quality looked right** on a synthetic test case: given two
  candidates where one had `amount_delta_after_fx=0.0` and one had a nonzero
  delta, the model correctly chose the zero-delta candidate and cited the right
  fact in its reasoning, at `confidence: 0.95`.
- **Neatlogs (see §6) genuinely captured the call** — full prompt, completion,
  token counts, and tags showed up in the local trace object with
  `"provider": "litellm"`, confirming LiteLLM calls to a custom `api_base` are
  traced, not just first-party provider SDKs.
- **Not yet verified:** accuracy at scale, UNSURE/NO_MATCH behavior on
  ambiguous or genuinely-unmatched cases, latency/cost at volume, and whether
  Neatlogs' hosted dashboard actually ingests the data (see §6 — the test run
  had no real Neatlogs API key, so server-side delivery is untested).

## 6. Neatlogs (observability) — three real gotchas found by testing, not assumed

Neatlogs traces every LLM call (prompt, response, cost, latency) for debugging
— it is **not** the accounting audit trail. The audit trail is the append-only
journal designed in ARCHITECTURE.md §2.8 (not built yet). Don't conflate them.

Found by reading the installed package source and by a live test run — not from
the docs site, which turned out to be unreliable (see below):

1. **docs.neatlogs.com documents an API that isn't published.** The docs show
   `neatlogs.wrap(client)` and `neatlogs.identify(session_id=..., end_user_id=...)`.
   The actual latest PyPI release (`neatlogs==1.1.8`) has neither. It works by
   **monkey-patching** `anthropic.Anthropic`, `litellm.completion`, etc. globally
   the moment `neatlogs.init(api_key=...)` runs — construct/call the client
   normally afterward, no wrapper needed. Before trusting any docs-site snippet,
   check `dir(neatlogs)` against what's actually installed.
2. **`neatlogs.add_tags(...)` (the module-level function) is broken in 1.1.8.**
   It always raises `RuntimeError: Tracker not initialized`, even right after a
   successful `init()`. Root cause, confirmed by reading both files:
   `neatlogs/__init__.py`'s `init()` sets its own module-global `_global_tracker`;
   `add_tags()` calls `core.get_tracker()`, which reads a **separate**
   module-global of the same name in `neatlogs/core.py` that `init()` never
   touches. **Workaround:** keep the object `init()` returns and call
   `.add_tags(...)` on that instance directly — verified working, and used in
   `llm_adjudicator.py`.
3. **Exporting to the hosted dashboard returned 404** (`POST
   https://app.neatlogs.com/api/data/v2` → `404 Client Error`) during the live
   test — but that test ran with an empty `NEATLOGS_API_KEY` (none was
   available). Re-test with a real key before concluding anything is actually
   broken server-side; this may simply be what an invalid/missing key does.

Also harmless, not a bug: LiteLLM logs repeated `"This model isn't mapped
yet"` warnings because it doesn't have pricing data for `glm-4-7-flash` — the
call still succeeds, but any `cost` field LiteLLM/Neatlogs reports for this
model is a meaningless default, not a real dollar figure.

## 7. Setup

```bash
pip install -U "neatlogs" litellm
cp .env.example .env   # fill in NEATLOGS_API_KEY and TENSORMUX_API_KEY
python -X utf8 llm_adjudicator.py
```

See [.env.example](.env.example) for the exact variables needed. Never commit
`.env` (already gitignored) or paste real API keys into chat/commits/issues —
if a key has ever appeared in plaintext anywhere outside your own `.env` file,
treat it as compromised and rotate it.

## 8. Repo / hosting

Pushed to [github.com/ambyKnight/reconFX](https://github.com/ambyKnight/reconFX),
`main` branch. Local git remote is named `github`.

## 9. File map

```
AGENTS.md              <- you are here
PROJECT.md              the pitch: what problem, why it's hard, evidence caveat
PIPELINE.md              measured findings + a 10-stage pipeline design, from BenchRec
ARCHITECTURE.md          the intercompany-specific system: pass loop, LLM adjudicator,
                         suspense/open-items, close reporting, feature backlog
GLOSSARY.md              every term used above, defined from scratch with real numbers
README.md               (repo placeholder from initial GitHub creation)

match.py                 deterministic matcher -- evidence-gathering, BenchRec
assemble.py              stage 2 group assembly -- bounded subset-sum that counts
                         rival solutions rather than stopping at the first
calibrate.py             stage 5 confidence -- Wilson lower bound, cross-fitted;
                         replaces hand-typed confidence literals
test_assemble.py         tests for both (26, stdlib-only, no pytest needed)
benchrec_probe.py        \
rule_archaeology.py       > evidence-gathering scripts, BenchRec, see PIPELINE.md
error_autopsy.py         /
sanity.py                \
truth_shape.py           / data-integrity checks on the BenchRec solution file

llm_adjudicator.py       the one built piece of the intercompany pass loop --
                         LLM verdict step (ARCHITECTURE.md §3), TensorMux + LiteLLM,
                         Neatlogs-traced

.env.example             required env vars (copy to .env, never commit .env)
.gitignore
```

`predictions.csv`, `match_group_profile.csv`, `error_autopsy.csv` are generated
by the evidence-gathering scripts above and are gitignored — regenerate them by
running the corresponding `.py` file, don't expect them to be in the repo.

## 10. What's next (in order)

From ARCHITECTURE.md §8, unchanged by anything above:

0. **Narrow Stage 1 blocking so group assembly can actually run** — currently
   99.9% ABSTAIN on real data (§3a). Everything about group targets is blocked
   behind this, and it is now the highest-value open item.
1. Mutual exclusivity + the pass loop (no model needed)
2. ~~Fix the collision-rule bug in `match.py`~~ — **done**, fitted on train, 76.2%
   at 99.89% (§3a)
3. Open-items subsystem with case files and re-attempt
4. ~~Structure/group assembly (subset-sum)~~ — **built and unit-tested**
   (`assemble.py`), blocked on item 0 for measurement; netting reconstruction
   still outstanding
5. Wire `llm_adjudicator.py` into the real pass loop, with the calibration and
   decision gates from ARCHITECTURE.md §3.3 (arithmetic re-verification,
   confidence recalibration, same threshold as every other tier) — right now
   it runs standalone against a fabricated dossier, not real leftover entries
6. Consolidation close report
7. Entity-pair memory, override learning, sampling assurance -- and, in
   parallel, keep looking for a real (or carefully-built synthetic)
   intercompany dataset to re-measure everything in this repo against
