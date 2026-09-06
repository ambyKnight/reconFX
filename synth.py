"""
Synthetic dataset harness: the model writes the texture, Python writes the truth.

    python synth.py --n 200 --seed 7 --out data/synth_v1.json          # no model
    python synth.py --n 200 --seed 7 --out data/synth_v1.json --live   # with model

Why a harness rather than "ask the model for a dataset"
--------------------------------------------------------
Asking a model for a labelled dataset produces labels only as good as the model,
and a matcher measured against them is really being measured against the model's
mistakes. So the boundary here is strict and one-directional:

    the model returns   -> names, reference formats, memo phrasing, habits
    Python returns      -> every amount, every date, every truth row

`_sanitise_texture` enforces it at the boundary: only strings are read out of the
model's reply, and anything numeric it volunteers is discarded rather than
trusted. Then `synth_schema.validate` re-derives every arithmetic claim before a
file is written, so a generator bug cannot become a wrong label.

The consequence worth stating: **the dataset is reproducible from its seed
alone.** Same seed, same amounts, same groupings, same answer key — with or
without a model, on any machine. Texture varies; truth cannot.

What it generates, and why those cases
--------------------------------------
Real cash application is not a stream of exact ties. The generated mix includes
short payments, fees taken in transit, overpayments, partials, unapplied cash,
and payments carrying no usable remittance text at all — which is the most cited
real-world cause of unapplied cash, and the case a clean generator would never
produce.

The case that matters most here is `GROUP_WITH_DECOY`: a rival set of invoices
that *also* sums to the payment. It is built explicitly, so the dataset contains
a known number of genuine ambiguities. That is what makes `assemble.py`'s rival
counting measurable — and measuring it is exactly what BenchRec could not
support, because its remitter field is a single constant (AGENTS.md §3b). Every
payment here carries a real `remitter_account`.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from synth_schema import KINDS, SCHEMA_VERSION, summarise, validate

# Mix of case kinds. Weighted toward the awkward ones on purpose: a generator
# that is 90% exact singles produces a matcher that looks excellent and is
# useless, because the easy cases were never the problem.
DEFAULT_MIX = {
    "EXACT_SINGLE": 30, "GROUP": 22, "GROUP_WITH_DECOY": 12, "FEE_DEDUCTED": 8,
    "SHORT_PAY": 7, "OVERPAYMENT": 4, "PARTIAL": 6, "NO_REMITTANCE": 7,
    "NO_MATCH": 4,
}

TEXTURE_TOOL = {
    "type": "function",
    "function": {
        "name": "record_counterparties",
        "description": "Record a set of fictional trading counterparties and how "
                       "each one writes payment references.",
        "parameters": {
            "type": "object",
            "properties": {
                "counterparties": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string",
                                     "description": "Fictional company name."},
                            "reference_style": {
                                "type": "string",
                                "description": "How this company formats invoice "
                                "references, with {n} for the number. e.g. "
                                "'INV-{n}', '{n}/24', 'Ref {n}'."},
                            "remittance_habit": {
                                "type": "string", "enum": ["always", "sometimes", "never"],
                                "description": "How reliably they quote invoice "
                                "numbers when paying. Real payers vary; include "
                                "some that never do."},
                            "memo_template": {
                                "type": "string",
                                "description": "How their payment memo reads, with "
                                "{refs} where invoice references go. e.g. "
                                "'PAYMENT {refs}', 'BACS {refs} THANKS'."},
                        },
                        "required": ["name", "reference_style", "remittance_habit",
                                     "memo_template"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["counterparties"],
            "additionalProperties": False,
        },
    },
}

TEXTURE_PROMPT = """Invent fictional trading counterparties for a synthetic \
accounts-receivable dataset used to test a cash reconciliation system.

Make them varied and realistic in the way that makes reconciliation hard:
- different reference formats, not one house style
- some that always quote invoice numbers when paying, some that sometimes do, \
and some that never do
- memo text that reads like real bank narrative: abbreviated, upper case, \
sometimes cluttered with words that carry no information

Do not invent amounts, dates, or invoice numbers -- those are generated \
separately. Only the names and the writing style. Call record_counterparties once."""

FALLBACK_TEXTURE = [
    {"name": "Northwind Trading Ltd", "reference_style": "INV-{n}",
     "remittance_habit": "always", "memo_template": "PAYMENT {refs}"},
    {"name": "Harbour Logistics", "reference_style": "{n}/24",
     "remittance_habit": "sometimes", "memo_template": "BACS {refs}"},
    {"name": "Pinewood Supplies", "reference_style": "REF{n}",
     "remittance_habit": "never", "memo_template": "FASTER PAYMENT"},
    {"name": "Alder & Co", "reference_style": "A{n}",
     "remittance_habit": "sometimes", "memo_template": "TRF {refs} THANKS"},
    {"name": "Cormorant Media", "reference_style": "CM-{n}",
     "remittance_habit": "always", "memo_template": "REMIT {refs}"},
]


def _sanitise_texture(raw: list[dict]) -> list[dict]:
    """Keep only strings the model was asked for; drop everything else.

    The boundary that keeps the model out of the answer key. If it volunteers an
    amount, a date, or an extra field, it is discarded here rather than flowing
    into the dataset, so a chatty or confused reply degrades the *prose* and
    nothing else. Malformed entries are skipped, not repaired: a half-understood
    reply is not worth guessing at.
    """
    habits = {"always", "sometimes", "never"}
    clean = []
    for row in raw or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        style = str(row.get("reference_style", "")).strip()
        memo = str(row.get("memo_template", "")).strip()
        habit = str(row.get("remittance_habit", "")).strip().lower()
        if not (name and "{n}" in style and memo):
            continue
        clean.append({"name": name, "reference_style": style,
                      "memo_template": memo,
                      "remittance_habit": habit if habit in habits else "sometimes"})
    return clean


def _texture_once(n: int) -> list[dict]:
    """One model call, sanitised. Raises on transport failure."""
    import os

    import litellm
    from llm_adjudicator import API_BASE, MODEL

    response = litellm.completion(
        model=MODEL, api_base=API_BASE,
        api_key=os.environ["TENSORMUX_API_KEY"],
        max_tokens=2000, temperature=1.0,
        messages=[{"role": "user",
                   "content": TEXTURE_PROMPT + f"\n\nProduce {n} counterparties."}],
        tools=[TEXTURE_TOOL],
        tool_choice={"type": "function",
                     "function": {"name": "record_counterparties"}},
    )
    out: list[dict] = []
    for call in getattr(response.choices[0].message, "tool_calls", None) or []:
        out.extend(_sanitise_texture(
            json.loads(call.function.arguments).get("counterparties")))
    return out


def fetch_texture(n: int, attempts: int = 3) -> tuple[list[dict], str, str | None]:
    """Ask the model for counterparty texture, retrying, then top up from fixtures.

    Returns `(texture, source, model)` where source is "llm", "mixed" or
    "fallback".

    Retries because this model's structured output is genuinely flaky: three
    identical calls measured 8, then 0, then 7 usable entries. The zero was a
    well-formed tool call whose contents did not survive `_sanitise_texture` —
    so the failure is not a transport error and nothing raises. An earlier
    version treated one bad draw as "no model available" and silently produced a
    fallback dataset while reporting success, which is exactly the kind of quiet
    substitution that makes a later result impossible to interpret.

    Results accumulate across attempts and are topped up from fixtures if still
    short, so a partial reply degrades the *proportion* of model-written text
    rather than discarding all of it. `source` says honestly which happened —
    "mixed" is not a failure, but it is not "llm" either, and a dataset should
    not claim more provenance than it has.
    """
    collected: list[dict] = []
    model_name = None
    try:
        import env
        env.load_env()
        env.require("TENSORMUX_API_KEY")
        from llm_adjudicator import MODEL
        model_name = MODEL

        for attempt in range(attempts):
            try:
                collected.extend(_texture_once(n - len(collected)))
            except Exception as exc:
                print(f"  texture: attempt {attempt + 1} failed "
                      f"({type(exc).__name__}), retrying")
                continue
            if len(collected) >= n:
                break
            print(f"  texture: attempt {attempt + 1} yielded "
                  f"{len(collected)}/{n} usable, retrying")
    except Exception as exc:
        print(f"  texture: model unavailable ({type(exc).__name__}), using fixtures")

    # De-duplicate by name; a retried call often repeats itself.
    seen, unique = set(), []
    for row in collected:
        if row["name"].lower() not in seen:
            seen.add(row["name"].lower())
            unique.append(row)

    if not unique:
        return list(FALLBACK_TEXTURE), "fallback", None
    if len(unique) >= n:
        return unique[:n], "llm", model_name
    topped = unique + [f for f in FALLBACK_TEXTURE][: n - len(unique)]
    print(f"  texture: {len(unique)} from model, {len(topped) - len(unique)} from fixtures")
    return topped, "mixed", model_name


# --- deterministic construction ----------------------------------------------

def build_counterparties(texture: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Counterparties, each with the remitter account BenchRec lacked."""
    parties = []
    for i in range(n):
        style = texture[i % len(texture)]
        parties.append({
            "id": f"CP{i:03d}",
            "name": style["name"] if i < len(texture) else f"{style['name']} {i}",
            # A real payer account: the key that makes group assembly solvable.
            "account": f"{rng.randint(10, 99)}-{rng.randint(10, 99)}-{rng.randint(10, 99)}"
                       f"/{rng.randint(10_000_000, 99_999_999)}",
            "reference_style": style["reference_style"],
            "remittance_habit": style["remittance_habit"],
            "memo_template": style["memo_template"],
        })
    return parties


def build_invoice(party: dict, seq: int, currency: str, issue: date,
                  amount_cents: int) -> dict:
    """One invoice. `allocation` mirrors the shape real ledgers use — currency,
    date, account, reference — which is also what makes prefix clustering
    (blocking.py) meaningful on this data."""
    reference = party["reference_style"].replace("{n}", str(100_000 + seq))
    return {
        "id": f"INV{seq:05d}",
        "counterparty_id": party["id"],
        "allocation": f"{currency}_{issue.isoformat()}_{party['account']}_{reference}",
        "amount_cents": amount_cents,
        "currency": currency,
        "issue_date": issue.isoformat(),
        "due_date": (issue + timedelta(days=30)).isoformat(),
        "reference": reference,
    }


def _memo(party: dict, invoices: list[dict], kind: str, rng: random.Random) -> str:
    """Payment narrative, following the counterparty's habit.

    Whether the memo carries usable references is a property of the payer, not
    of the case — which is what makes reference corroboration a *sometimes*
    available second signal rather than a reliable one. A generator where the
    memo always helps would make the matcher look far better than it is.
    """
    habit = party["remittance_habit"]
    quote = (kind != "NO_REMITTANCE") and (
        habit == "always" or (habit == "sometimes" and rng.random() < 0.5))
    refs = " ".join(i["reference"] for i in invoices) if quote else ""
    template = next((t for t in [party.get("memo_template")] if t), "PAYMENT {refs}")
    return " ".join(template.replace("{refs}", refs).split()) or "PAYMENT"


def _residual_for(kind: str, total: int, rng: random.Random) -> int:
    """The exact, intended difference between payment and invoices."""
    if kind == "FEE_DEDUCTED":
        return -rng.choice([250, 350, 500, 1200])          # bank fee in transit
    if kind == "SHORT_PAY":
        return -max(100, int(total * rng.uniform(0.01, 0.08)))  # deduction/dispute
    if kind == "OVERPAYMENT":
        return rng.choice([100, 500, 1000, 2500])
    if kind == "PARTIAL":
        return -int(total * rng.uniform(0.3, 0.7))
    return 0


def _pick_kinds(n: int, mix: dict[str, int], rng: random.Random) -> list[str]:
    """Draw `n` case kinds from the weighted mix, then shuffle.

    Drawn proportionally rather than sampled independently, so a small dataset
    still contains every kind. A 50-payment set that happened to include zero
    decoys would silently not test the thing decoys exist to test.
    """
    kinds: list[str] = []
    total_weight = sum(mix.values())
    for kind, weight in mix.items():
        kinds.extend([kind] * max(1, round(n * weight / total_weight)))
    rng.shuffle(kinds)
    if len(kinds) >= n:
        return kinds[:n]
    return kinds + rng.choices(list(mix), k=n - len(kinds))


def _group_size(kind: str, rng: random.Random) -> int:
    """How many invoices this case settles.

    Sizes follow the shape measured on BenchRec (AGENTS.md 3a): size 2 dominates
    at ~52%, and the tail past 5 is thin. A flat distribution would
    over-represent the expensive cases and misrepresent how often the cheap
    subset-sum path actually applies.
    """
    if kind in ("EXACT_SINGLE", "PARTIAL", "SHORT_PAY", "OVERPAYMENT", "FEE_DEDUCTED"):
        return 1
    return rng.choices([2, 3, 4, 5, 6], weights=[52, 16, 8, 6, 3])[0]


def _amount(rng: random.Random) -> int:
    """An invoice amount in cents.

    Round numbers are deliberately common: they are what make coincidental rival
    sums happen in real ledgers, and a generator drawing only from a uniform
    range would produce far fewer accidental collisions than reality does.
    """
    if rng.random() < 0.3:
        return rng.choice([25_000, 50_000, 75_000, 100_000, 150_000, 250_000])
    return rng.randint(1_500, 500_000)


class _Builder:
    """Accumulates invoices, payments and truth for one dataset.

    A small class rather than threaded counters: sequence numbers, the RNG and
    the three output lists all have to stay in step, and passing four mutable
    things between free functions is how off-by-one bugs get into an answer key.
    """

    def __init__(self, parties: list[dict], currency: str, start: date,
                 rng: random.Random):
        self.parties = parties
        self.currency = currency
        self.start = start
        self.rng = rng
        self.invoices: list[dict] = []
        self.payments: list[dict] = []
        self.truth: list[dict] = []
        self.seq = 0

    def _new_invoice(self, party: dict, issue: date, amount: int) -> dict:
        self.seq += 1
        invoice = build_invoice(party, self.seq, self.currency, issue, amount)
        self.invoices.append(invoice)
        return invoice

    def add_case(self, kind: str, index: int) -> None:
        party = self.rng.choice(self.parties)
        issue = self.start + timedelta(days=self.rng.randint(0, 40))
        value_date = issue + timedelta(days=self.rng.randint(1, 12))
        payment_id = "PAY%05d" % index

        if kind == "NO_MATCH":
            # Unapplied cash: money arrived, nothing in the ledger explains it.
            self._emit(payment_id, party, value_date, _amount(self.rng), [], kind, 0)
            return

        chosen = [self._new_invoice(party, issue, _amount(self.rng))
                  for _ in range(_group_size(kind, self.rng))]
        total = sum(i["amount_cents"] for i in chosen)
        residual = _residual_for(kind, total, self.rng)

        if kind == "GROUP_WITH_DECOY":
            self._add_decoy(party, issue, total)

        self._emit(payment_id, party, value_date, total + residual, chosen,
                   kind, residual)

    def _add_decoy(self, party: dict, issue: date, total: int) -> None:
        """A rival pair summing to the same total, left unclaimed.

        The case the whole uniqueness argument turns on: two different sets of
        invoices that both explain the payment exactly. A matcher reporting
        UNIQUE here is wrong, and with this the error is measurable rather than
        argued about.
        """
        split = self.rng.randint(int(total * 0.25), int(total * 0.75))
        self._new_invoice(party, issue, split)
        self._new_invoice(party, issue, total - split)

    def _emit(self, payment_id: str, party: dict, value_date: date, amount: int,
              invoices: list[dict], kind: str, residual: int) -> None:
        self.payments.append({
            "id": payment_id,
            "counterparty_id": party["id"],
            "amount_cents": amount,
            "currency": self.currency,
            "value_date": value_date.isoformat(),
            # The key BenchRec lacks -- see synth_schema's module docstring.
            "remitter_account": party["account"],
            "remitter_name": party["name"],
            "memo": _memo(party, invoices, kind, self.rng),
        })
        self.truth.append({
            "payment_id": payment_id,
            "invoice_ids": [i["id"] for i in invoices],
            "kind": kind,
            "residual_cents": residual,
            "note": "",
        })

    def add_distractors(self, n: int) -> None:
        """Open invoices nobody paid.

        Without these, every candidate belongs to some answer and blocking looks
        far more precise than it is. Real ledgers are mostly unpaid invoices.
        """
        for _ in range(n):
            party = self.rng.choice(self.parties)
            issue = self.start + timedelta(days=self.rng.randint(0, 40))
            self._new_invoice(party, issue, _amount(self.rng))


def generate(seed: int = 7, n_payments: int = 200, n_parties: int = 12,
             currency: str = "GBP", live: bool = False,
             mix: dict[str, int] | None = None) -> dict:
    """Build a complete, validated dataset. Deterministic given `seed`."""
    rng = random.Random(seed)
    if live:
        texture, source, model = fetch_texture(n_parties)
    else:
        texture, source, model = list(FALLBACK_TEXTURE), "fallback", None

    parties = build_counterparties(texture, n_parties, rng)
    builder = _Builder(parties, currency, date(2026, 1, 6), rng)
    for index, kind in enumerate(_pick_kinds(n_payments, mix or DEFAULT_MIX, rng)):
        builder.add_case(kind, index)
    builder.add_distractors(n_payments // 2)

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "currency": currency,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "texture_source": source,
            "model": model,
            "n_payments": len(builder.payments),
            "n_invoices": len(builder.invoices),
        },
        "counterparties": parties,
        "invoices": builder.invoices,
        "payments": builder.payments,
        "truth": builder.truth,
    }


def write(dataset: dict, path: str | Path) -> Path:
    """Validate, then write.

    A dataset failing the oracle is never persisted: an invalid answer key on
    disk outlives the run that produced it and quietly poisons every measurement
    taken against it afterwards.
    """
    problems = validate(dataset)
    if problems:
        raise ValueError("dataset failed validation:\n  " + "\n  ".join(problems[:20]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic reconciliation dataset.")
    parser.add_argument("--n", type=int, default=200, help="payments to generate")
    parser.add_argument("--parties", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--currency", default="GBP")
    parser.add_argument("--out", default="data/synth_v1.json")
    parser.add_argument("--live", action="store_true",
                        help="ask the model for counterparty texture")
    args = parser.parse_args()

    dataset = generate(args.seed, args.n, args.parties, args.currency, args.live)
    path = write(dataset, args.out)
    print(summarise(dataset))
    print("\nvalidated, wrote %s" % path)


if __name__ == "__main__":
    main()
