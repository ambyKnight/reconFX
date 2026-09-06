# Architecture — intercompany reconciliation that works like a controller

Successor to [PIPELINE.md](PIPELINE.md). That document described a *pipeline*:
data enters one end, decisions come out the other. This one describes something
closer to how the job is actually done — repeated passes, an open-items ledger,
and a consolidation close report at the end.

Terms are defined in [GLOSSARY.md](GLOSSARY.md).

> **A note on where the numbers come from.** This system is designed for
> **intercompany reconciliation** — getting two entities of the same group to
> agree on what they owe each other. No public, labelled intercompany dataset
> exists (see PROJECT.md for why), so every measured number in this document —
> pool sizes, precision, cost — is carried over from **BenchRec**, the bank-vs-
> ledger cash reconciliation dataset used as the evidence base in PIPELINE.md.
> They are cited here as *sizing estimates for the analogous mechanism*, not as
> measurements of intercompany data. Where a number appears, it is BenchRec's;
> where the architecture appears, it is designed for intercompany specifically.
> Re-measuring against real intercompany data, once available, is the
> outstanding validation step this whole document is built to make cheap to run.

---

## 1. The organizing idea: passes, not a pipeline

A controller does not process every intercompany entry across the group in one
sweep. They do the obvious ones first, and then — this is the important part —
**the obvious ones make the hard ones easier.**

If entry #4,001 on the UK entity's books certainly pairs with entry #7,882 on
the US entity's books, both are spent. Neither can be pulled into some other
pairing. A three-way tie that looked unbreakable at the start of the day
resolves itself for free once two of the three candidates have been claimed by
other entries.

A matcher that scores every entry **independently**, in one pass, and never
revisits, can tell two different UK entries "you pair with US entry #7,882" and
never notice. That is not a small bug — it means a chunk of what looks like
genuine ambiguity is really just entries judged before their rivals were
resolved.

So the architecture is a **loop**, and each turn is cheaper and smarter than the
last:

**Pass 1 — Certainties.** Deterministic rules only, no LLM. Exact amount (after
FX translation at the transaction's own rate) plus same or adjacent period,
collapsing to one counterpart entry. Also the recognisable **netting-center
settlements** — one lump sum against many small cross-charges, which look like
noise but are structurally unambiguous once you know the netting convention.
These are consumed and removed from the pool.

**Pass 2 — Re-block after consumption.** Rebuild candidate pools now that pass
1's entries are spent. Free — no new logic, no model. Some former ties are now
single-candidate. Re-run pass 1's rules on the survivors.

**Pass 3 — Known-adjustment matching.** Apply the *documented* transfer-pricing
markup and standard FX-translation tolerance for each entity pair, and re-test:
an entry that "doesn't tie" by exactly the group's known 8% intercompany markup,
or by exactly a currency-translation rounding difference, isn't ambiguous — it's
correct, and the rule should recognise it as such rather than send it to a
human every month. Consume, loop back to pass 2.

**Pass 4 — LLM adjudication.** Only now, on what deterministic logic could not
settle, and only on a pool that earlier passes have already thinned and
disambiguated.

**Pass 5 — LLM structure proposal.** The exotic tail: an IC loan drawdown split
across several ledger entries on one side, a recharge that was netted on one
side but not the other, "this looks like the same cost pool booked under two
different intercompany account codes." The LLM proposes a *structure*,
deterministic code verifies the arithmetic and the FX translation.

**Stop** when a pass resolves nothing new. Everything still unresolved becomes
an open intercompany item, with a record of what was tried.

Two things follow from this design and are worth saying explicitly:

**Matching is a global assignment problem, not a per-row lookup.** Each pass
should resolve mutually-exclusive claims rather than let two entries on one
side claim the same entry on the other. Greedy-by-confidence (highest-confidence
claims first, consume, repeat) is the controller's own method and is good
enough to start. Min-cost bipartite matching, run separately per entity pair, is
the formal version if greedy proves lossy.

**Every pass makes the next one cheaper.** Passes 1–3 cost nothing but CPU, and
each one removes entries from the LLM's bill. Ordering the loop this way is not
just tidier — it is the cost control.

---

## 2. The components

Nine, each with one job. Passes are orchestrated over these; the components
themselves are stateless except where noted.

**2.1 Ingest.** Canonical, typed representation of every entity's sub-ledger.
Money as `Decimal`, never float — and every amount carries **both** its
transaction currency and its functional currency, since the same intercompany
entry legitimately looks like two different numbers on two different books.
Emits a run manifest (data hash, code version, config hash, FX-rate-table
version, model version) so any close can be reproduced exactly.

**2.2 Candidate engine (blocking).** Cheap, recall-oriented shortlisting per
entity pair: amount within FX-translation tolerance, shared intercompany
reference or cost-allocation code, counterpart entity + period window. Its job
is to never lose the right answer; precision is somebody else's problem. It
reruns each pass against the shrinking pool.

**2.3 Deterministic matcher.** The rules that need no model: exact ties, known
markups, known netting conventions. Fast, free, explainable, and the fallback if
anything downstream misbehaves.

**2.4 Structure assembler.** Bounded subset-sum and netting reconstruction, with
uniqueness as the signal: one subset summing to the target (after FX and markup
adjustment) is evidence; several is ambiguity, and it says so rather than
picking.

**2.5 LLM adjudicator.** Covered in §3.

**2.6 Decision and risk control.** Takes a calibrated probability and returns
AUTO / REVIEW / OPEN ITEM, against a threshold that is fitted rather than
chosen. **Crucially, this component is the only thing that can post or clear.**
The LLM proposes; this decides. That separation is what keeps the audit story
coherent — auditors specifically test intercompany reconciliation controls, and
"a model decided" is not a control.

**2.7 Open-items subsystem.** §4. This is the part most systems treat as a
leftovers bin and it deserves better — for intercompany, this is also the
subledger an auditor samples directly.

**2.8 Journal.** Append-only record of every decision: what, why, how sure, by
which model version, when, and by whose authority. Reversals are new rows,
never edits. Every row is tagged with **both** entities in the pair.

**2.9 Reporting.** §5. The consolidation close pack.

---

## 3. The LLM's job, precisely

The value of an LLM here is real but narrow. It is good at weighing scruffy
human-written evidence — a cost-allocation memo, a recharge description that
reads "Q2 SHARED ENG COST ALLOC PER SLA 2024-03" — and saying what that
probably means, and at reasoning across a markup or an FX rate that a rigid rule
wasn't written to expect. It is bad at arithmetic, and it is a security surface.
So the job description is deliberately tight.

### 3.1 What it receives

A **dossier**, assembled deterministically: the entry, the shortlist of
counterpart candidates on the other entity's books (numbered), and for each
candidate a set of **pre-computed facts** — amount delta after FX translation at
the period's own rate, whether that delta matches this entity pair's known
transfer-pricing markup, period/close-calendar gap, shared reference or
cost-allocation code, whether this entity pair historically settles via a
netting center, whether the candidate is already claimed.

The model never computes any of these. It reads them.

### 3.2 What it returns

A structured verdict, schema-enforced:

- **MATCH** — candidate index *n* (an index into the list it was given, never an
  identifier it composed itself)
- **MATCH_GROUP** — a set of indices (e.g. one netted settlement against several
  individual cross-charges)
- **NO_MATCH** — nothing here is the counterpart; this is a genuine open
  intercompany item, and one entity likely owes the other a booking
- **UNSURE** — I cannot tell

plus a confidence, a reason in one plain sentence, and which facts drove it.

**UNSURE and NO_MATCH are the point.** A model that must choose between MATCH
and NO_MATCH will confabulate a winner, and you have rebuilt the exact problem
this project exists to solve — software that sounds equally certain whether it
is right or wrong. For intercompany specifically, a confabulated match is worse
than for bank reconciliation: it can misstate a number that is disclosed to
auditors and regulators as a related-party balance. The abstention options are
load-bearing, and the prompt should make abstaining explicitly respectable.

### 3.3 What happens to the verdict

Three gates, in order:

1. **Arithmetic and FX re-verification.** If the model says these three entries,
   translated at the period's FX rate, sum to the counterpart amount, Python
   redoes the translation and the sum. Disagreement voids the verdict. The model
   is never trusted on a number, and it is especially never trusted to apply an
   FX rate.
2. **Calibration.** The model's stated confidence is meaningless raw — a
   self-reported "90%" is a linguistic habit, not a probability. Note that the
   Anthropic API does not expose token logprobs, so there is no free confidence
   signal to read out. You get calibrated numbers one of two ways: fit isotonic
   regression on labelled outcomes to map stated confidence onto observed
   accuracy, or sample the model *k* times and use agreement rate as the
   confidence (self-consistency). Agreement is usually better calibrated and
   costs *k*×. Until real intercompany outcomes exist to calibrate against, this
   step has to run on whatever labelled evidence is available (BenchRec, or an
   initial hand-reviewed batch) and should be treated as provisional until
   re-fitted on real intercompany history.
3. **The same threshold as everything else.** The calibrated LLM verdict
   competes on identical terms with the deterministic tiers. It gets no special
   authority because it produced prose.

This is the architectural point worth holding onto: **the LLM is one more scorer
feeding the existing calibration and decision stages.** It does not get a side
door to the ledger. Everything that made the rule-based system trustworthy
applies to it unchanged.

### 3.4 Why this pool suits an LLM particularly well

On the BenchRec evidence base (see §-note above), roughly half of what a
one-shot matcher calls "ambiguous" turns out on inspection to be a *group*
waiting to be assembled — one payment covering several invoices — not a genuine
contest between rivals. Intercompany data structurally has more of this, not
less: netting centers exist specifically to bundle many small cross-charges into
one settlement, and IC loan drawdowns routinely get split across several ledger
lines on one side while landing as a single balance on the other. Reading a
netting memo or a cost-allocation description to work out which underlying
entries a lump sum covers is a reading-comprehension problem, which is exactly
what a language model is for and exactly what an amount-and-date rule cannot do.

Similarly, in a no-candidate pool a meaningful share of items will have no real
counterpart at all — one entity genuinely forgot to book the other half.
**NO_MATCH being a first-class, sometimes-correct answer matters at least as
much here as it did for cash reconciliation**, because in intercompany the two
sides are two departments of the same company: a true structural gap ("go raise
this entry") is a routine, expected outcome, not a rare edge case.

### 3.5 Cost

Sizing carried over from the BenchRec evidence base: at a leftover pool of a few
thousand entries with a double-digit average candidate count, one full pass on
Claude Opus 5 runs on the order of **$0.02 per item**, roughly half that through
the Batch API — and intercompany reconciliation is inherently a scheduled,
overnight job, so the batch discount is close to free money. Three-sample
self-consistency roughly triples the bill. The real intercompany pool size will
differ (fewer entity pairs than a bank has counterparties, but more of them
recur with the same partner month after month, which the caching strategy below
should exploit directly).

Cache the instruction block, schema, and each entity pair's *standing
information* (its markup convention, its usual netting pattern, its FX
translation basis) — these repeat identically every month for the same pair, so
they belong ahead of the cache boundary; only the current period's dossier
varies. And passes 1–3 shrink the bill before it is ever incurred.

At these numbers, **cost is not a design constraint** — a group spends far more
on the controller hours this replaces. Spend the tokens on reasoning quality and
self-consistency, not on trimming prompts.

### 3.6 Where the LLM does not belong

- **Not in blocking.** Deterministic keys handle candidate generation in
  seconds at any scale a group's intercompany volume will reach.
- **Not doing arithmetic, and especially not FX translation.** Ever.
- **Not making the post decision.** That is a calibrated threshold with a
  guarantee behind it.
- **Not as the system of record.** The journal is. Store the verdict the model
  gave; never re-derive it later by asking again. Models are non-deterministic
  and get upgraded — last quarter's consolidation close must still reproduce
  exactly, and the only way to guarantee that is to have written the answer
  down.

---

## 4. Open items, properly

The open-items ledger is where the honesty lives. It is not a failure bucket; it
is the system's considered statement that it does not know — and for
intercompany, it is also literally the schedule an external auditor will ask to
see and sample from at period end.

### 4.1 Every open item carries a case file

Not just "unmatched" — the record should say: which entities and currencies are
involved, which passes ran, what was tried, the best candidates considered and
why each was rejected, the model's reasoning if it got that far, the amount and
its materiality, the age, and the assigned category.

A controller picking up an open item should never have to redo the
investigation the system already did — and neither should the auditor reviewing
the same item three months later.

### 4.2 Categories, because they route differently

- **Timing difference** — the counterpart entity hasn't closed the period yet,
  or books on a different cut-off date. Expected to clear on its own next
  period.
- **Unbooked counterpart** — one side has genuinely not recorded its half.
  Someone at that entity must raise the entry.
- **FX translation variance** — matched except for a difference explained by
  rate timing or rounding. Resolvable by a tolerance rule.
- **Transfer-pricing markup mismatch** — the delta doesn't match this pair's
  documented markup. Needs someone who knows the pricing agreement, not just the
  numbers.
- **Netting/settlement mismatch** — a lump sum doesn't reconcile to the
  cross-charges it's supposed to cover.
- **Genuine ambiguity** — several plausible counterparts, none separable. Needs
  a human who knows the business relationship between the two entities.
- **Suspected duplicate** — the same cross-charge booked twice, on one or both
  sides.
- **Data error** — malformed, wrong sign, wrong entity code, impossible period.

The category determines *who* gets it and *what they do* — often a specific
person at a specific entity, since intercompany resolution frequently requires
someone on the *other* side of the pair to act, not just the person running the
reconciliation.

### 4.3 Aging

Bucket by days outstanding: 0–7, 8–30, 31–60, 60+. Aging is the health metric of
the whole system, and for intercompany it carries extra weight: a persistent,
aged intercompany imbalance is a specific, well-known item auditors test for,
because it can indicate a control weakness in how the group's entities talk to
each other, not just a matching failure.

### 4.4 Open items are re-attempted, not archived

**Every run re-feeds open items through the full pass loop.** This is not an
optimisation — it is the single most valuable property of the subsystem.

An entry booked by the UK entity on the 28th, before the US entity closes its
own books on the 2nd, is unmatchable in month one and trivially matchable in
month two once both sides have posted. Timing differences are the largest
category of real-world intercompany open items, precisely because entities on
different close calendars are a normal, permanent feature of a multinational
group — and they resolve themselves *if you keep looking*. A system that files
open items away permanently guarantees that a human will eventually do work the
machine could have done for free.

Auto-clearing an open item on a later run is a first-class event: journalled,
reported, and reversible.

### 4.5 Materiality

Controllers do not spend forty minutes chasing a rounding difference of one
currency unit. Below a configurable threshold, the system should propose a
**write-off** rather than an investigation, batched for a single approval. Above
a high-value threshold — and intercompany balances between major entities can be
very large — the opposite: escalate immediately, and demand higher confidence
before ever auto-clearing (see §6.3).

---

## 5. The consolidation close report

After each pass — and at period end — the system produces a pack. This is the
deliverable a controller actually signs off to the group's consolidation team,
so it should be built as a first-class output rather than scraped out of logs.

**5.1 Reconciliation summary.** Volume and value, **by entity pair**: entries
in, entries matched, % matched by count *and by value* (they differ, often
sharply). Auto-cleared vs. human-reviewed vs. open. Comparison against prior
periods.

**5.2 Open-items register.** Every open item by category, entity pair, and
currency, aged, sorted by value, each with its case file. This is the working
document for the close, not an appendix — and the document handed to an auditor
who asks for the intercompany reconciliation support.

**5.3 Elimination readiness report.** Specific to consolidation: which
intercompany balances are confirmed matched and safe to eliminate in the
consolidated financials, which are still open and therefore represent a
consolidation risk, and the **net out-of-balance amount by entity pair** — the
single number a consolidation team actually needs before they can close the
group's books.

**5.4 Control and assurance report.** The part that makes an auditor
comfortable: how many entries were auto-cleared, at what calibrated confidence,
the measured precision on the evidence base, the calibration evidence, and the
results of the sampling programme (§6.4). This is where a match-rate number
becomes an attestable statement rather than a slide — and where the honest
caveat from PROJECT.md belongs until real intercompany precision is measured.

**5.5 Override log.** Every case where a human disagreed with the system —
overrode a match, rejected a proposal, cleared something it called ambiguous.
This is both a control signal and the highest-quality training data available,
since each one is a labelled example produced by an expert for free — and, for
intercompany, it is the beginning of the real dataset this project doesn't yet
have.

**5.6 Drift and trend.** Match rate, open-item aging, category mix, and
per-entity-pair performance over time. A newly onboarded entity, a changed
transfer-pricing agreement, a migrated ERP at one subsidiary — any of these
degrade quality quietly, and the numbers move before anyone notices the cause.

**5.7 The narrative.** A good use of the LLM: write the plain-English summary
that goes on top. *"Match rate for the UK–US pair held at 84% this period. Open
items with the Singapore entity rose 30%, driven almost entirely by 12 entries
following their ERP migration on the 14th — reference format changed and the
matcher hasn't adapted yet."* The model writes prose over numbers computed
deterministically — it never produces a figure of its own.

---

## 6. Further features, roughly by value

### 6.1 Entity-pair behavioural memory

Learn each pair's habits: UK→US recharges settle net-30 via the EMEA netting
center, always carry an 8% markup, reference cost pools as `CC-####`; the
Singapore entity's close lags the group calendar by three business days. Store
it, feed it to both the feature model and the LLM dossier.

This is what makes a controller fast — they *know* their group's entities — and
it converts a large class of fuzzy matches into confident ones. Probably the
highest-value item on this list, and it compounds: the longer the system runs
against a given group, the better it gets at that group specifically.

### 6.2 Learning from overrides

Each human correction is a labelled example. Feed them back into the scorer and
the calibration, and track whether the categories humans keep correcting are
shrinking. For intercompany specifically, this override log is also the closest
thing to real training data the project will have until a licensed dataset
exists — treat it accordingly and store it well from day one.

### 6.3 Risk-weighted thresholds

A rounding-difference entry and a multi-million-unit IC loan settlement should
not face the same confidence bar. Make the threshold a function of exposure:
auto-clear small items freely, demand near-certainty on large ones, and let
expected cost of error — not a single global number — drive the decision.

This mirrors how controllers actually allocate attention, and it usually buys
match rate *and* reduces risk simultaneously, since most volume is small and
most risk is concentrated in a few large intercompany balances.

### 6.4 Continuous assurance sampling

In production there is no labelled solution file. So how do you know the
precision bar still holds?

Randomly sample a slice of auto-cleared items for human verification,
continuously. That gives a live, unbiased precision estimate with no ground
truth — and it is the honest answer to the question an auditor will certainly
ask about an intercompany control. Pair it with a **circuit breaker**: if
sampled precision drops below the bar, stop auto-clearing and route everything
to humans until someone looks.

### 6.5 Duplicate cross-charge detection

Same entity pair, same amount, near dates, no distinguishing reference. Real
money — the same shared cost gets double-booked across entities more often than
groups would like to admit, and intercompany reconciliation is exactly where it
surfaces. This feature can pay for the entire project on its own.

### 6.6 Transfer-pricing anomaly flags

Reconciliation is a natural place to catch a pricing agreement being applied
inconsistently. Worth flagging: a markup that suddenly differs from the pair's
historical norm, a cross-charge with no supporting allocation memo, a new
account code appearing between two entities that never traded before. Not
accusations — flags for a human, and useful input to the group's transfer-pricing
documentation, which tax authorities scrutinise directly.

### 6.7 What-if / shadow mode

Run a proposed config or model change against historical closes and report what
would have changed: which entries would newly auto-clear, which would stop,
which past decisions would have flipped. Nothing that touches consolidated
numbers should ever be deployed without this.

### 6.8 Batch review by pattern

Cluster the review queue by shape rather than presenting it as a flat list.
"These 30 items are all the UK–US pair, all off by exactly the Q2 rate — accept
all?" One decision instead of thirty. This is where reviewer hours actually go.

### 6.9 Queue ordering by value × uncertainty

Order the human queue so the most consequential and least certain items come
first. If the close runs out of time — it always does — the work that got done
should be the work that mattered most to the consolidated numbers.

### 6.10 Run continuously, not monthly

Nightly rather than at period-end. Open items never accumulate into a wall,
timing differences resolve within days, and the close becomes a report rather
than a scramble — which matters more for intercompany than for cash
reconciliation, since more entities means more independent close calendars
colliding at month-end.

### 6.11 Period lock and correct-forward

Once a period closes, its journal is immutable. Later corrections post to the
current period with a reference to the original. Standard accounting practice,
and it constrains the data model, so design it in early.

### 6.12 Segregation of duties, per entity

The system proposes; a named human at **each** entity in the pair approves;
both identities are recorded against the entry. For intercompany specifically,
segregation matters on both sides of the pairing, not just one — a single
approver clearing both halves of an intercompany entry is itself a control gap
auditors look for.

---

## 7. Guardrails specific to LLM-plus-money

**Reference and memo fields are entered by whoever books the entry — including,
in some flows, someone at the counterparty entity.** That text reaches a prompt
in a system that can clear or post intercompany balances. Three mitigations, all
structural rather than prompt-based: the model may only select from a
pre-computed candidate list **by index**; it can never emit an identifier of its
own; and its output schema admits no free-form action, only a verdict. Treat
memo and reference text as data in the dossier, clearly delimited, and never as
instruction.

**Nothing posts or clears on the model's say-so.** Arithmetic and FX translation
are re-verified, confidence is recalibrated, and the decision stage owns the
final call.

**Reproducibility beats re-derivation.** Store the verdict and the model
version. Never reconstruct a past consolidation close by asking the model
again.

**Cost ceiling per run**, with the run degrading to "everything unresolved
becomes an open item" rather than silently truncating.

**A deterministic fallback path.** If the LLM layer is unavailable, the system
still reconciles — worse, but correctly, with a smaller auto-clear rate and a
larger queue. Never a hard dependency, since intercompany closes run on a fixed
group calendar that cannot simply wait for a vendor outage.

---

## 8. What to build first

The pass loop and the open-items subsystem are the structural pieces; the LLM is
comparatively easy to add once the structure holds it correctly. The evidence
base stays BenchRec until real intercompany data exists — every step below is
buildable and testable against it today.

1. **Mutual exclusivity + the pass loop** (passes 1–2). No model, no new
   algorithms, resolves an unknown but real slice of the ambiguous pool for
   free, provable on the evidence base first.
2. **Fix the collision rule** (PIPELINE.md §1.1) as a general lesson —
   collapsing candidates before judging ambiguity, and treating a crowded
   amount+date/amount+period bucket as evidence *for* a batch settlement rather
   than against it.
3. **Open-items subsystem with case files and re-attempt**, including the
   intercompany-specific categories in §4.2. Delivers the "it's honest about
   what it doesn't know" promise, which is the actual product.
4. **Structure assembly** (pass 3): netting reconstruction and transfer-pricing-
   aware matching. This is the piece with no BenchRec analogue and the one most
   worth prototyping against synthetic intercompany scenarios early.
5. **LLM adjudicator** (pass 4) on what survives, with calibration and the three
   gates.
6. **Consolidation close report.** Makes all of the above legible to the person
   who signs.
7. **Entity-pair memory, override learning, sampling assurance** — and, in
   parallel, the search for a real (or a carefully-built synthetic) intercompany
   dataset to re-measure everything against.

The ordering is deliberate: every step before the LLM makes the LLM's job
smaller, cheaper, and more accurate — and every step is honest, throughout,
about which parts are proven and which are designed-but-unmeasured.
