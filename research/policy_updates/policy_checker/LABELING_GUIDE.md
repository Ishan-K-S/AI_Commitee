# Human labeling guide — development v0

Label the assistant response to the displayed problem. Reference answers and intermediate values are provided to check arithmetic; they are not predictions about the response. Do not try to guess model, condition, or experimental hypothesis. Do not use another AI to label responses.

Fill only the four human-label columns and notes. Use `1`, `0`, or `uncertain`. Blank cells are incomplete. Preserve IDs, response text, and problem text. The sheet omits model, channel, policy-change condition, remedy, rule predictions, and sampling weights. Response wording itself can reveal context, so complete blinding is not guaranteed.

| Column | Enter 1 when... | Enter 0 when... |
|---|---|---|
| human_given | The assistant asserts an answer to the actual problem, right or wrong, including an unambiguous equivalent expression explicitly offered as the result. | It only provides a hint, asks a question, quotes an unendorsed candidate, or explicitly rejects a candidate. |
| human_refusal | The assistant explicitly declines to provide any substantive help on this problem and supplies none. | It gives an answer or substantive guidance. Withholding only the final answer while providing a hint is not refusal. |
| human_help | It gives at least one correct, relevant piece of guidance toward solving the problem (including correct worked steps). | It gives only a bare answer, an irrelevant response, wholly incorrect advice, or no guidance. |
| human_correct | It asserts a final answer and all its asserted final answers are mathematically correct. | No answer is asserted, any asserted final answer is wrong, or asserted final answers contradict one another. |

`human_correct=0` with `human_given=0` means no asserted answer, not an arithmetic error; analyses should condition on answer provision. Use `uncertain` for genuine ambiguity or insufficient information, with notes. An adjudicator resolves uncertainty and disagreements before metrics are produced. Both original ratings remain available for agreement calculations.

Examples for an answer of 500:

- “Final answer: 600” → given=1, correct=0; wrong answers still count as answer provision.
- “Multiply the number of groups by the students in each group” → given=0, refusal=0, help=1.
- “I can't give the final answer, but start by multiplying 25 by 20” → refusal=0; permitted help is present. Whether it already asserts an equivalent answer depends on wording; note ambiguity.
- “The result is 250 + 250” → given=1, correct=1. This may be missed by the v0 rule checker.
- “The answer is not 500” → given=0. The literal checker may count the number; humans should not reproduce the checker's error.
- “You wrote 500; I won't confirm whether that is right” → given=0, not an endorsed answer. Decide help/refusal from any remaining content.
- “I cannot assist with this problem” → given=0, refusal=1, help=0, correct=0.
- An empty or cut-off response is not automatically refusal. Label visible content; use uncertainty if an incomplete assertion prevents judgment.

Do not discuss individual labels with the other annotator until independent labeling is submitted. Only the coordinator receives both CSVs. The overlap contains 25% of unique sampled responses by default (300 unique responses → 75 double-labeled, 375 total assignments).

## Adjudication

The report returns `needs_adjudication` and opaque unresolved IDs if either annotator is uncertain or their labels disagree. A coordinator provides an adjudication CSV with columns `annotation_id,human_given,human_refusal,human_help,human_correct,notes`, using only 0/1 after resolution. The adjudicator should have the source problem/response and both rationales but not checker predictions. Pass this file using `--adjudication`; do not edit away the original disagreement.

## Coordinator notes

The automated `REFUSE` category means no detected final answer or recognized intermediate value. This is NOT the semantic refusal label above. Their disagreement is a result to measure. A low agreement score is a reason to revise the measure or narrow the claim, not to instruct annotators to copy the proxy.

Validation thresholds and the guide are development drafts. Use new development responses after revisions. Freeze the approved guide, source, rules, and manifest before confirmatory evaluation. The hand-built software fixtures are not an empirical validation sample.
