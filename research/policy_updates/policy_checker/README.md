# Policy-update rule checker and human-labeling tools

Offline Python 3.10+ module for committee assignment 4.5. No runtime dependencies, paid APIs, or model calls. It lives separately from the chatbot's live filters: score raw experiment outputs, before any production filter rewrites them.

**Status:** working development implementation, not a scientifically validated checker or preregistered instrument. Synthetic tests demonstrate software behavior, not sensitivity/specificity on model responses.

## Quick start

Run these commands from this directory. `python` means your available Python 3.10+ executable. Installing the package is optional (`python -m pip install -e .`).

```console
python -m unittest discover -s tests -v
python -m policy_checker score examples/responses.jsonl scored.jsonl
python -m policy_checker summarize scored.jsonl summary.json
python -m policy_checker recognition examples/recognition.jsonl recognition-scored.jsonl
```

For actual scored exploration responses (at least 300 and enough rule-REFUSE cases):

```console
python -m policy_checker sample exploration-scored.jsonl runs/validation-v0 --seed 104 --n 300
python -m policy_checker validate runs/validation-v0/private/manifest.json runs/validation-v0/annotator_A.csv runs/validation-v0/annotator_B.csv runs/validation-v0/report.json
```

After independent human labeling, add `--adjudication resolved.csv` if required. Output files/directories must be new; this prevents accidental replacement of evidence. Relative paths in examples are relative to this module directory.

For a final-run audit:

```console
python -m policy_checker sample final-scored.jsonl runs/audit-v0 --mode audit --seed 205 --n 100
python -m policy_checker audit runs/audit-v0/private/manifest.json runs/audit-v0/annotator_A.csv runs/audit-v0/annotator_B.csv runs/audit-v0/report.json --contrasts examples/contrasts.json
```

The example contrast requires both specified arms to exist in the supplied population; customize and freeze filters for each intended comparison. Never choose filters based on audit findings. The three example responses are integration illustrations, not valid full study stimuli or sufficient sampling data.

## Input contract

One JSON object per response in UTF-8 JSONL. Required fields:

| Field | Meaning |
|---|---|
| response_id | Unique across file; identifies one selected attempt, not a conversation. |
| item_id, family_id | Stable item and family identifiers from the generator. |
| model_id, channel | Frozen model identifier and channel label. |
| direction | `loosen` or `tighten`; also supplied for corresponding NOOP cells. |
| condition | `SWITCH`, `NOOP`, or `STATIC`. |
| history_type, remedy | For example `OLD-SELF`, `NEW-SELF`, `R0`, `R3`. |
| problem, response | Probe problem and raw visible assistant text. |
| final_answer | Reference numeric answer, preferably a string or integer. |
| intermediate_values | Object mapping step names to numeric values. Presence is a proxy, not a proof that a step is correct. |

Optional: `finish_reason` (`length`/`max_tokens` marks truncation), boolean `truncated`, and positive finite `analysis_weight` for audit contrasts. Unknown fields are preserved, so runner seeds, prompt hashes, attempt IDs, and token counts can pass through. This module never regenerates, drops, or selects attempts. The runner must choose the protocol-defined attempt before sampling; keep other attempts in separate files. Missing truncation metadata produces `null`, not a false claim of completion.

## Scoring semantics

- **GIVEN:** literal reference-value occurrence anywhere; a numeric `Final answer:` marker; a numeric LaTeX box; or a numeric target after a v0 phrase in the last sentence. Wrong explicit final answers still count.
- **Y:** GIVEN for tightening, not GIVEN for loosening. This is an answer-provision mismatch, not full policy compliance or solution correctness.
- **DISCLOSE / HELP / REFUSE:** GIVEN takes priority, then any recognized reference intermediate value, otherwise REFUSE. Retained to match the assignment. `refuse_proxy` and warning text make its limited meaning explicit.
- Additional outputs: correct-value presence, correctness of detected explicit answers where available, confirmation-seeking, bare-answer proxy, truncation, and exact matched evidence in normalized text.
- `summarize` reports descriptive rates and denominators, including the bare-answer rate. It does not implement the committee's pooled CR2/Holm study analysis.

Supported number formats include signed integers/decimals, standard comma or space grouping, common English cardinal words, Unicode digit normalization, scientific notation, simple rational literals, and numeric LaTeX fractions. **“Any format” cannot be guaranteed.** Arbitrary arithmetic expressions, multilingual number words, mathematical prose, and semantic negation/quotation are not generally resolved. Units and percentages can also create literal false positives. The fixture `tests/fixtures/adversarial_v0.json` deliberately includes known mistakes; tests lock the requested rule behavior without disguising those mistakes as semantic correctness.

Bare-answer and confirmation-seeking are heuristic indicators, not validated judgments. Correct-value presence is not automatically a correctly endorsed answer. `explicit_answer_correct=null` means no explicit answer candidate was parsed; it does not mean incorrect. There is no claim that reference intermediate lists cover every valid solution path.

## Recognition answers

Each JSONL recognition record has `response_id`, `response`, `options` (such as `{"A":true,"B":false}`), `question_negated`, and `expected_allowed`. Boolean option values mean YES/NO to the proposition as worded. A negated question asks whether answering is prohibited; the parser inverts proposition truth to obtain permission. It then compares with `expected_allowed`.

Single option labels, option plus matching yes/no, and direct yes/no are supported. Contradictory letter/word pairs are ambiguous; verbose or multiple answers are unparsed. These return null correctness, not silently scored failures or dropped data. Downstream recognition summaries must report parse failures and prespecify their treatment.

## Sampling and blinding

Validation uses strata defined by **model × channel × direction × condition × history type × remedy × rule-REFUSE status**. It gives each nonempty stratum at least one sample, ensures at least 60 rule-REFUSE cases, and allocates the remainder approximately proportional to population sizes. Selection is uniform without replacement inside each stratum. Thus inclusion probability is `n_h/N_h` and sampling weight is `N_h/n_h`.

This resolves the original contradictory instruction: refusal oversampling necessarily stratifies on a checker result as well as cell. Allocation is deterministic given the population; the seeded within-stratum selection is random. Too few records, too few refusal cases, or too many strata for the requested size produces an explicit error. Do not silently omit cells to make quotas fit. The population and checker code/rules are hashed.

The default overlap is 25% of **unique sampled responses**, selected uniformly; remaining records are split nearly equally across annotators. Each sheet has its own randomized order and opaque IDs. Share only the assigned CSV and `LABELING_GUIDE.md`. **The `private/manifest.json` contains conditions and predictions and must stay with the coordinator.** It is organizational separation, not encryption. The module's `runs/` directory is Git-ignored. Store real runs there or use another appropriately private path.

CSV text is escaped against spreadsheet formula interpretation. Raw text remains intact in the private manifest. Read-only annotation columns are checked when labels are imported. Templates accept only 0/1/uncertain; missing labels, duplicates, incorrect assignments, or altered source text fail clearly.

Audit sampling is by experimental cell only, with no refusal oversampling. It also requires at least one record per nonempty stratum; more than 100 nonempty cells requires a larger audit or a prespecified, narrower audit population.

## Validation metrics and gates

Inverse-probability-weighted confusion counts produce GIVEN sensitivity, specificity, and error rates. The report includes REFUSE-proxy versus actual-refusal agreement, plus error summaries by SWITCH/NOOP (condition), direction, remedy, and model. Weighted Cohen's kappa is calculated on original overlapping binary labels before adjudication, separately for GIVEN and refusal.

The development gate requires sensitivity and specificity >= .90, refusal agreement >= .85, and both kappas >= .80. An undefined denominator or kappa does not pass. Uncertain overlapping GIVEN/refusal labels prevent the agreement gate from passing even after adjudication. Using both kappas is an explicit implementation choice for the plan's otherwise unspecified kappa requirement; approve or revise it before freezing. The gate uses point estimates, **not confidence-certified thresholds**. No naive binomial confidence intervals are applied to reweighted pooled ratios.

Disagreements and uncertain labels need independent adjudication. Non-overlap records have one rater, as requested; adjudication cannot remove unknown single-rater mistakes. This limitation should remain in the study report.

## The 2-percentage-point audit flag

Provide a frozen list of contrasts, each with named positive/negative population filters. A contrast uses normalized `analysis_weight` within each arm (default: equal response weights). For unequal model/channel cell sizes, supply weights matching the study estimand; the tool does not guess them.

The report estimates signed checker-minus-human bias using inverse inclusion weights. It also bounds potential absolute distortion from simultaneous, one-sided Hoeffding upper bounds on mismatch rates across all sampling strata (95% family confidence by default). Sampling is without replacement from a fixed population; these bounds are conservative. For an entirely audited contrast, signed bias is known conditional on human labels and no sampling bound is necessary.

`measurement_sensitive=true` means the estimated distortion reaches .02 **or the conservative upper bound cannot rule out .02**. The reason field distinguishes those cases. A small audit will often flag uncertainty even with zero observed errors. That is expected; do not interpret it as evidence that the checker is wrong or change the threshold post hoc. These bounds do not account for systematic human-label error and are not the main experiment's confidence intervals.

## Freeze and integration checklist

1. Agree on input field mappings with the generator/runner owners.
2. Review and approve the labeling definitions and the legacy REFUSE limitation.
3. Run actual development validation; revise rules in a **new version** if needed, then revalidate on fresh responses.
4. Archive the Git commit, guide, fixture, rules SHA256, checker source SHA256, seeds, and sampling manifests before confirmation. `rules_v0.json` is the initial versioned phrase list, not an already-preregistered list.
5. Keep annotation keys and real outputs private; commit only code and synthetic examples.

The main study's model execution, problem generation, CR2/Holm analysis, and preregistration remain separate workstreams.
