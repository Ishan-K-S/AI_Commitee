# Assignment 4.5 fit review

Reviewed October 1, 2026. Scope: checker, human-labeling workflow, and measurement audit for the policy-update experiment. This is an implementation review; it is not empirical validation against real model responses.

| Requirement | Implementation / verification | Remaining boundary |
|---|---|---|
| GIVEN evidence, Y by direction | Literal/marker/box/last-sentence rules, evidence spans; tests for both directions and NOOP | Not all equivalent mathematical formats; not full solution compliance |
| HELP and REFUSE | Priority-ordered legacy categories; explicit `refuse_proxy` flag | Conceptual help can be mislabeled; human refusal is independently labeled |
| Confirmation, bare answers, truncation | Per-response flags and descriptive cell rates | Heuristics; missing truncation metadata stays unknown |
| Recognition polarity/order | Explicit option mappings, negation reversal, rejection of conflicting/unparseable answers | Runner owns the three-replicate recognition aggregation and parse-failure policy |
| Versioned phrases/adversarial cases | Versioned v0 JSON, rules/code hashes, hand-authored expected-rule and human labels | v0 is a development snapshot; approve and freeze before preregistration |
| Validation sample of 300, REFUSE >=60 | Cell-by-proxy strata, probability sampling, exact inclusion weights | Quota is impossible if population has fewer than 60 proxy cases; fails explicitly |
| Blinded sheets, 25% overlap | Separate private key, randomized opaque IDs, 75 shared items / 375 assignments | Content can reveal context; coordinator must not share private manifest |
| Sensitivity, specificity, refusal agreement, kappa | Weighted metrics, pre-adjudication agreement, unresolved-label gating, condition splits | Point-estimate thresholds; both given/refusal kappas required, an explicit design choice |
| Audit sample of 100 | Cell-stratified sampling without refusal enrichment | Many cells mean only 1–2 examples each; cannot certify small bias |
| Flag potential 2-point distortion | Contrast-specific signed error estimate and conservative simultaneous bounds | Typically flags uncertainty with small audits; does not prove distortion |
| Study-level relevance | Audit supports Y, GIVEN, and refusal-harm contrasts; full 80-cell layout test | Main paired CR2/Holm analysis belongs to a different workstream |

## Gaps found and fixed in this review

- The first audit implementation only handled answer-provision mismatch (Y). It now also supports GIVEN and the refusal proxy, including the remedy's refusal-harm check.
- Numeric word parsing missed a valid cardinal number following a prose conjunction, such as “And five hundred students remain.” Added a regression test and fix.
- Sampling now validates incoming record fields and normalizes supplied analysis weights to numeric values.
- The measurement-sensitive flag now explicitly fires on either estimated distortion or uncertainty reaching the threshold, matching its documented meaning.

## Readiness judgment

Suitable for development integration and an actual labeling pilot. Not ready to call a confirmatory validated measurement instrument. The next required evidence is performance on independently labeled model outputs, especially differences between SWITCH/NOOP and R0/R3.

The literal policy endpoint intentionally follows the task specification. Passing its software tests does not resolve known construct problems: “not 500” can trigger correct-value occurrence; an equivalent expression can be missed; purely conceptual help can become proxy REFUSE. The adversarial fixtures make these limitations visible. If human validation fails, revise the operational definition and version it, rather than adjusting human labels to match the checker.

Use the same frozen record format with the generator and runner. They must provide valid reference values, final-attempt selection, direction/condition metadata, and appropriate pooling weights. The tools preserve raw responses and never alter histories or query models.
