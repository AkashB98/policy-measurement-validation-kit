# policy-measurement-validation-kit

**A validation kit for LLM-assisted policy-outcome measurement that refuses to publish a headline rate when the annotator is quietly failing one subgroup: the overall error on the seeded data is 0.0775 — comfortably publishable — while the planted `sarcasm` subgroup's error is 0.242424 against a 0.15 floor, so the gate returns REFUSE for that reason alone.**

When a team uses an LLM-assisted annotator to measure a social outcome at scale — what share of items carry a policy-relevant outcome, say — the number that gets published is only as good as the annotator's errors. The dangerous failure is not a bad annotator; it is an annotator that is good *on average* and bad somewhere specific, because the average is what everyone looks at. This kit rebuilds the validation stage end to end on synthetic data: a human gold label set is loaded as given, the annotator's **sensitivity** (share of gold positives it catches) and **specificity** (share of gold negatives it leaves alone) are estimated overall and per subgroup, the headline rate is corrected for measurement error with the **Rogan-Gladen** estimator, drift between two annotation batches is checked, and a **publish gate** returns PUBLISH or REFUSE with reasons — and nothing more. The build was motivated by an OpenAI Applied Scientist, Product Policy Research & Measurement role (2026-11-15) centred on LLM-assisted annotation validation and measurement-error checks on social-outcome measures, with light backing from a Gusto governance role (2026-11-11).

> **Not the golden-set repo.** How a trustworthy human gold set is built — rater agreement (Krippendorff's alpha), adjudication of disagreed cases, leakage checks — is the sibling `human-feedback-golden-set-kit`'s job, and this kit deliberately contains **no inter-annotator agreement work**: the gold set arrives as given, and this kit validates the *machine* annotator against it. It also contains **no evidence bundle or audit trail** — that is `model-governance-evidence-kit` territory. The gate returns publish/refuse + reasons only.

> **Synthetic data only.** Every item, gold label, and annotation is fabricated by a seeded generator about a fictional policy-outcome program. No real person, post, or company is used anywhere. The annotator is simulated with a planted error structure (see below) so the machinery can be checked against known answers.

Stdlib-only Python — deliberately. A measurement validation is a trust artifact: anyone should be able to read every line, re-run offline, and get byte-identical outputs. Zero network, zero API keys, zero external dependencies.

## The seeded dataset at a glance (seed 42)

2,400 items × 2 batches, true prevalence 0.30 in every subgroup. Planted annotator: good on `standard` (planted sensitivity/specificity 0.94/0.96) and `dialect` (0.93/0.95), much worse on the planted subgroup `sarcasm` (0.70/0.78), and slightly worse in batch 2 than batch 1 (drift). A `clean` control variant uses one uniformly good annotator (0.93/0.95) everywhere.

| Measure (planted variant) | Value | Reading |
|---|---|---|
| True rate (gold) | **0.301667** | Known only because the data is synthetic |
| Naive rate (annotated) | **0.315833** | What an unvalidated pipeline would publish; abs. error **0.014167** |
| Rogan-Gladen corrected rate | **0.301667** | Exact — see the identity note below; this is algebra, not a result |
| Annotator sensitivity / specificity (pooled) | **0.895028 / 0.934368** | Estimated against the gold set |
| Overall error rate | **0.0775** | Under the 0.10 publish floor — looks fine |
| Subgroup error: `standard` / `dialect` / `sarcasm` | **0.044510 / 0.071523 / 0.242424** | `sarcasm` breaches its 0.15 floor by a wide margin |
| Drift statistic (batch 2 − batch 1 error) | **0.005** | Under the 0.05 floor; per-subgroup drift up to 0.025166 (`dialect`) |
| Gate verdict | **REFUSE** | Exactly one reason: the `sarcasm` subgroup |
| Clean control | overall error **0.054583**, worst subgroup **0.070707**, drift **0.015833** | Gate **PUBLISH**es |

**The transport diagnostic is the honest correction number.** Error rates estimated on batch 1 only, applied to batch 2's naive rate (0.310833): corrected **0.290587** against batch 2's true 0.300833 — absolute error **0.010246**, versus the naive rate's own error of **0.010000**. At this seed the transported correction does *not* beat doing nothing; it is reported as measured (see Development loop).

## POC framing

This kit is the validation stage of a proof-of-concept measurement pipeline: *can an LLM-assisted annotator's headline rate be trusted enough to publish?* The POC succeeds or fails on stated criteria, not vibes:

**Success criteria**
1. The annotator's sensitivity and specificity are estimable against the gold set — overall and for every subgroup (a slice with no gold positives or negatives cannot be validated, and the gate refuses rather than guessing).
2. The corrected headline rate is computable (sensitivity + specificity > 1; an annotator at or below that line carries no information about the label).
3. Every gate check below passes on the data as measured.

**Thresholds, and the gate they feed**

| Check | Threshold (floor) | Gate input |
|---|---|---|
| Overall error rate | ≤ **0.10** | `checks.overall_error_rate` |
| Any subgroup error rate | ≤ **0.15** | `checks.subgroup_error_rates` (worst subgroup named) |
| Drift statistic \|err(batch 2) − err(batch 1)\| | ≤ **0.05** | `checks.drift_statistic` |

The gate returns **PUBLISH** only when all three hold; otherwise **REFUSE** with one reason per breached check. On the seeded planted data the POC *fails by design* — criterion 3 fails on the subgroup check — and on the clean control it passes. Both outcomes are pinned in the evals (E7, E10), so a POC run that silently changed either verdict would fail loudly.

## Quickstart

```bash
# 1. Generate the seeded datasets (planted = headline, clean = control)
python3 cli.py generate --variant planted --out planted.jsonl
python3 cli.py generate --variant clean --out clean.jsonl

# 2. Data-quality validation of a dataset file
python3 cli.py validate --input planted.jsonl

# 3. Full report: rates + correction, per-subgroup errors, drift, gate
python3 cli.py report --input planted.jsonl
python3 cli.py report --variant clean

# 4. CI gate: exit 0 = PUBLISH, 1 = REFUSE (EXPECTED for the seeded
#    planted dataset — it refuses by design, via the subgroup),
#    2 = IO / data-quality error
python3 cli.py check --input planted.jsonl
python3 cli.py check --input clean.jsonl    # exit 0
python3 cli.py check                        # seeded planted default: exit 1

# 5. One-command end-to-end demo
python3 demo.py

# 6. Tiny localhost API: GET /report, GET /health on 127.0.0.1
python3 cli.py serve --port 8765
```

## How it works

**Error estimation (`measure.py`).** Every annotated label is compared with gold: sensitivity = TP/(TP+FN), specificity = TN/(TN+FP), error rate = (FP+FN)/N — computed pooled, per subgroup, per batch, and per subgroup×batch cell. On the seeded planted data the pooled sensitivity/specificity (0.895028/0.934368) hide the split that matters: `sarcasm` sits at 0.642857/0.802817 while `standard` is at 0.946860/0.959315.

**Rogan-Gladen correction (`measure.py`).** If the apparent (annotated) prevalence is *p* and the annotator's sensitivity/specificity are *Se*/*Sp*, the prevalence consistent with those error rates is *(p + Sp − 1)/(Se + Sp − 1)*, clamped to [0, 1]; *Se + Sp ≤ 1* raises, because no correction exists for an uninformative annotator. **Identity note:** when *Se*/*Sp* are estimated on the same rows the headline is computed from, the corrected rate equals the gold rate *exactly* — pure algebra (the apparent rate is built from the same confusion counts). The kit therefore also reports the **transport** diagnostic above — error rates from batch 1 applied to batch 2 — which is the version whose accuracy is actually earned, and which drift can break.

**Drift (`measure.py`).** The drift statistic is stated plainly: |error_rate(batch 2) − error_rate(batch 1)|, pooled, with the same quantity reported per subgroup. Pooling dilutes: the planted batch-2 degradation shows up as 0.021637–0.025166 inside subgroups but only 0.005 pooled — one more reason the gate judges subgroups on their own numbers.

**Publish gate (`measure.py`).** Three floors (table above) plus an estimability requirement; the verdict is PUBLISH or REFUSE with reasons, and nothing more — no bundle, no trail, no score. The headline test is pinned at two levels: a hand-built unit case where overall error sits exactly at its floor (0.10, not above) while one subgroup breaches, and the seeded dataset (eval E7), where the refusal carries exactly one reason and it names `sarcasm`.

## Architecture

| Module | Job |
|---|---|
| `generator.py` | Deterministic synthetic data: 2,400 items, planted error structure (good annotator, failing `sarcasm` subgroup, batch-2 drift) plus a clean control variant; JSONL/JSON/CSV load + write |
| `validation.py` | Data-quality gate (fields, label/batch domains, duplicate ids, both batches and ≥ 2 subgroups present) feeding exit code 2 |
| `measure.py` | Confusion counts, sensitivity/specificity/error per slice, Rogan-Gladen correction + transport diagnostic, drift check, publish gate with stated floors |
| `report.py` | One deterministic report document: dataset summary, rates, errors, drift, gate |
| `cli.py` | `generate` / `validate` / `report` / `check` (CI exit 0/1/2) / `demo` / `serve` (stdlib `http.server` on 127.0.0.1, GET /report + /health) |
| `demo.py` | End-to-end narrative with asserts: determinism, rates + transport, the subgroup headline, drift, gate refuse + publish, check exit codes, live serve round-trip |
| `fixtures/` | The two seeded datasets (`sample_planted.jsonl`, `sample_clean.jsonl`), pinned against the generator by eval E14 |

## Evals

`python3 evals/run_evals.py` → **15/15 golden evals**, report committed at `evals/eval_report.json` (md5 `db102a54d74ac120c89f5a8b394b36eb`, byte-identical across 3 verified runs):

- **E1** generator determinism — both variants' JSONL md5 pinned (`4e719e93…` / `a54108aa…`)
- **E2** shape pinned (2,400 rows; subgroups 1348/755/297; batches 1200/1200)
- **E3** hand-computed statistics (sensitivity 0.75, specificity 0.666667, error 0.3; Rogan-Gladen 0.34 → 0.30)
- **E4** headline rates pinned (true 0.301667, naive 0.315833, Se 0.895028, Sp 0.934368)
- **E5** in-sample identity pinned as a sanity fact (corrected == true exactly; naive error 0.014167)
- **E6** transport pinned (batch-2 corrected 0.290587, error 0.010246 vs naive 0.010000)
- **E7** HEADLINE gate test (overall 0.0775 under floor; `sarcasm` 0.242424 over floor; REFUSE, exactly one reason)
- **E8** per-subgroup errors and Se/Sp pinned
- **E9** drift pinned (batch errors 0.075/0.080, statistic 0.005, per-subgroup drift)
- **E10** clean control publishes (overall 0.054583, worst subgroup 0.070707, drift 0.015833)
- **E11** gate hand cases (subgroup-only refusal at overall exactly 0.10; drift-only refusal; clean grid publishes)
- **E12** CLI `check` exit codes 1 / 0 / 1 / 2 (bare `check` refuses by design on the seeded default)
- **E13** serve round-trip matches local computation exactly
- **E14** bundled fixtures pinned (files equal the seeded generator output)
- **E15** full report deterministic and internally consistent

Unit tests: `python3 -m unittest discover -s tests` → **57 hermetic tests**, including a no-network enforcement test (sockets stubbed), hand-computed confusion/sensitivity/Rogan-Gladen cases with the arithmetic in comments, clamping and uninformative-annotator edges, the headline subgroup-breach gate test, drift-only and overall-only refusals, custom floors, CLI exit codes, and serve round-trips.

## Development loop — what the checks actually caught

1. **The first correction demo was vacuous, and the numbers proved it.** With sensitivity/specificity estimated on the same 2,400 rows as the headline rate, the Rogan-Gladen corrected rate came out *exactly* equal to the gold rate (0.301667) on both variants — not approximately, exactly. That is an algebraic identity (the apparent rate and the error estimates are built from the same confusion counts), so presenting it as "the correction works" would have been rigged. The fix was not to fudge the design but to report both quantities for what they are: the pooled correction as a consistency identity, and a **transport** diagnostic — batch-1 error rates applied to batch 2 — as the honest accuracy number. The transport result at this seed is unflattering (error 0.010246 vs the naive rate's 0.010000) and is printed anyway.
2. **Pooling hides the drift it is supposed to catch.** The planted batch-2 degradation is real inside every subgroup (drift 0.008770–0.025166), yet the pooled drift statistic is only 0.005: each batch's pooled error is a mix-weighted average (subgroup share and gold prevalence differ between batches), so the per-subgroup degradations partially cancel in the pool. An early draft of the demo asserted pooled drift "shows the planted degradation"; it does not, at any honest reading. The report now carries per-subgroup drift next to the pooled statistic, and the demo asserts only what the numbers support: drift is present per subgroup and under the pooled floor — the refusal belongs to the subgroup check, not the drift check.
3. **The clean control's transported correction is worse than its naive rate** (error 0.010877 vs 0.003333): batch 2's naive rate happened to land close to truth, and batch 1's error-rate sampling noise is larger than that gap. Rather than re-seeding until the control flattered the method, the evals pin the transport numbers as measured and the honest-limits section states the general point: a transported correction adds the variance of the error estimates, so on small validation batches it can lose to the naive rate.

## Scope and honest limits

- **The correction assumes gold is truth.** Every sensitivity/specificity figure inherits whatever error lives in the gold set itself. This kit loads gold as given; establishing that gold deserves the name is `human-feedback-golden-set-kit`'s job, and a biased gold set would pass through here undetected.
- **The pooled correction is an identity in-sample** (see Development loop). It demonstrates the estimator is implemented consistently; it is not evidence about accuracy on new data. The transport diagnostic is the accuracy evidence, and at this seed it is a wash.
- **The pooled correction assumes one error structure.** With subgroup error rates as different as 0.044510 and 0.242424, a single pooled Se/Sp pair is the wrong model for any subgroup-level rate — which is precisely why the gate refuses instead of publishing a corrected number built on it. Per-subgroup correction is the obvious next step and is deliberately not shipped here: on the seeded data it would produce a publishable-looking number for a subgroup whose annotator the gate has already judged unfit.
- **Drift is one number per comparison.** |Δ error rate| between two batches catches level shifts; it does not attribute them (annotator change? item-mix change? both batches here share the generator's subgroup mix by construction, which real batches do not), and it says nothing about drift *within* a batch.
- **The simulated annotator is a caricature.** Planted per-cell sensitivity/specificity with independent errors is a test fixture, not a model of a real LLM annotator, whose errors correlate with item difficulty and with each other. Real error structure is messier and usually less legible.
- **The API is a demo surface.** It binds 127.0.0.1 only, holds its dataset in memory, and resets on restart.

## Why this project

It takes the step of an AI measurement program that decides whether anyone should believe the headline — validating the annotator against gold, per subgroup, across batches, behind a gate that can say no — and rebuilds it as an engineering artifact: an error estimator verified against hand-computed cases, a correction whose in-sample identity is stated instead of sold, a transport diagnostic reported even when it loses, and a refusal that fires on the subgroup everyone else would have averaged away. The average is not the measurement; the worst slice you are willing to publish is.
