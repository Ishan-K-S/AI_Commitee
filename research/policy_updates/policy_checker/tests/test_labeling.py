import csv
import json
from pathlib import Path
import tempfile
import unittest

from policy_checker.checker import Checker
from policy_checker.labeling import make_sample, export_sample, load_annotations, safe_cell
from policy_checker.metrics import audit_report, kappa, validation_report
from test_checker import record


def population(n=400):
    checker = Checker()
    return [checker.score(record("Hint" if i % 4 == 0 else "500" if i % 4 == 1 else "Use 25 groups.",
                                 response_id=f"r{i:04d}", item_id=f"i{i}",
                                 condition="SWITCH" if i % 2 else "NOOP")) for i in range(n)]


def synthetic_labels(manifest, sheets):
    lookup = {s["annotation_id"]: s for s in manifest["selected"]}
    ratings, gold = {}, {}
    for s in manifest["selected"]:
        aid = s["annotation_id"]
        score = s["record"]["scoring"]
        label = {"human_given": str(int(score["given"])), "human_refusal": str(int(score["refuse_proxy"])),
                 "human_help": "1", "human_correct": "1", "notes": "synthetic test, not human validation"}
        ratings[aid] = {name: label.copy() for name in s["annotators"]}
        gold[aid] = {f: int(v) for f, v in label.items() if f.startswith("human_")}
    return ratings, gold


class SamplingTests(unittest.TestCase):
    def test_full_eighty_cell_study_layout(self):
        checker, rows = Checker(), []
        cells = [("SWITCH", "OLD-SELF", "R0"), ("NOOP", "NEW-SELF", "R0"),
                 ("SWITCH", "NEW-SELF", "R0"), ("SWITCH", "OLD-SELF", "R3"),
                 ("NOOP", "NEW-SELF", "R3")]
        for model in range(4):
            for channel in ("C3", "C4"):
                for direction in ("loosen", "tighten"):
                    for condition, history, remedy in cells:
                        for item in range(15):
                            rows.append(checker.score(record("Hint" if item % 2 else "500",
                                response_id=f"r{len(rows)}", item_id=f"i{item}", model_id=f"m{model}",
                                channel=channel, direction=direction, condition=condition,
                                history_type=history, remedy=remedy)))
        validation, sheets = make_sample(rows, n=300, seed=101)
        audit, _ = make_sample(rows, n=100, seed=102, mode="audit")
        self.assertEqual(len(validation["strata"]), 160)
        self.assertEqual(len(audit["strata"]), 80)
        self.assertTrue(all(s["n"] >= 1 for s in validation["strata"].values()))
        self.assertTrue(all(s["n"] >= 1 for s in audit["strata"].values()))
        self.assertEqual(sum(map(len, sheets.values())), 375)

    def test_refusal_audit_catches_conceptual_help_error(self):
        checker = Checker()
        rows = [checker.score(record("Explain the concept." if i % 2 else "Use 25 groups.",
                  response_id=f"r{i}", direction="tighten", condition="NOOP",
                  history_type="NEW-SELF", remedy="R3" if i % 2 else "R0")) for i in range(20)]
        manifest, sheets = make_sample(rows, n=20, mode="audit")
        _, gold = synthetic_labels(manifest, sheets)
        for label in gold.values():
            label["human_refusal"] = 0
        contrast = {"name": "refusal-harm", "outcome": "refuse_proxy", "positive": {"remedy": "R3"}, "negative": {"remedy": "R0"}}
        report = audit_report(manifest, gold, [], [contrast])
        result = report["contrasts"][0]
        self.assertTrue(result["measurement_sensitive"])
        self.assertAlmostEqual(result["estimated_signed_measurement_bias"], 1)
        self.assertAlmostEqual(result["estimated_human_contrast"], 0)

    def test_loosen_audit_inverts_label_error_sign(self):
        checker = Checker()
        rows = [checker.score(record("500" if i % 2 else "Use 25 groups.", response_id=f"r{i}",
                                     direction="loosen", condition="SWITCH" if i % 2 else "NOOP")) for i in range(20)]
        manifest, sheets = make_sample(rows, n=20, mode="audit")
        _, gold = synthetic_labels(manifest, sheets)
        for label in gold.values():
            label["human_given"] = 0
        report = audit_report(manifest, gold, [], [{"name": "loosen", "positive": {"condition": "SWITCH"}, "negative": {"condition": "NOOP"}}])
        self.assertAlmostEqual(report["contrasts"][0]["estimated_signed_measurement_bias"], -1)
        self.assertAlmostEqual(report["contrasts"][0]["estimated_human_contrast"], 0)

    def test_forced_oversampling_recovers_population_totals(self):
        checker = Checker()
        rows = [checker.score(record("Hint" if i < 60 else "500", response_id=f"r{i:04d}")) for i in range(400)]
        manifest, _ = make_sample(rows, n=300, seed=5)
        selected = manifest["selected"]
        self.assertEqual(sum(s["record"]["scoring"]["refuse_proxy"] for s in selected), 60)
        self.assertAlmostEqual(sum(s["sampling_weight"] * s["record"]["scoring"]["refuse_proxy"] for s in selected), 60)
        self.assertAlmostEqual(sum(s["sampling_weight"] * s["record"]["scoring"]["given"] for s in selected), 340)

    def test_sampling_quota_overlap_weights_and_reproducibility(self):
        rows = population()
        manifest, sheets = make_sample(rows, seed=42)
        other, _ = make_sample(list(reversed(rows)), seed=42)
        self.assertEqual(manifest, other)
        self.assertEqual(len(manifest["selected"]), 300)
        self.assertEqual(manifest["overlap_count"], 75)
        self.assertEqual(sum(map(len, sheets.values())), 375)
        self.assertGreaterEqual(sum(s["record"]["scoring"]["refuse_proxy"] for s in manifest["selected"]), 60)
        self.assertAlmostEqual(sum(s["sampling_weight"] for s in manifest["selected"]), 400)
        for sheet in sheets.values():
            self.assertNotIn("condition", sheet[0])
            self.assertNotIn("scoring", sheet[0])
            self.assertNotIn("response_id", sheet[0])

    def test_quota_failure_is_explicit(self):
        with self.assertRaises(ValueError):
            make_sample(population(40), n=30, refuse_min=20)

    def test_audit_no_refusal_oversampling(self):
        manifest, _ = make_sample(population(), n=100, mode="audit")
        self.assertEqual(manifest["refuse_quota"], 0)
        self.assertEqual(len(manifest["strata"]), 2)

    def test_validation_complete_and_missing_class(self):
        manifest, sheets = make_sample(population(), seed=41)
        ratings, gold = synthetic_labels(manifest, sheets)
        report = validation_report(manifest, ratings, gold, [])
        self.assertTrue(report["gate_pass"])
        self.assertEqual(report["overall"]["given_sensitivity"], 1)
        self.assertEqual(report["kappa_pre_adjudication"]["human_given"], 1)
        self.assertIsNone(kappa([("1", "1", 1)] * 10))
        self.assertFalse(validation_report(manifest, ratings, gold, ["unresolved"])["gate_pass"])

    def test_disagreement_requires_adjudication(self):
        manifest, sheets = make_sample(population(), seed=41)
        ratings, _ = synthetic_labels(manifest, sheets)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "labels"
            export_sample(out, manifest, sheets)
            for name in ("A", "B"):
                path = out / f"annotator_{name}.csv"
                with path.open(encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    fields, rows = reader.fieldnames, list(reader)
                for row in rows:
                    row.update(ratings[row["annotation_id"]][name])
                with path.open("w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fields)
                    writer.writeheader()
                    writer.writerows(rows)
            _, gold, unresolved = load_annotations(manifest, {n: out / f"annotator_{n}.csv" for n in ("A", "B")})
            self.assertEqual(len(gold), 300)
            self.assertFalse(unresolved)
            with self.assertRaises(FileExistsError):
                export_sample(out, manifest, sheets)

    def test_csv_formula_protection(self):
        self.assertEqual(safe_cell("=SUM(A1)"), "'=SUM(A1)")
        self.assertEqual(safe_cell("normal"), "normal")

    def test_small_audit_cannot_certify_two_points(self):
        manifest, sheets = make_sample(population(), n=100, mode="audit")
        _, gold = synthetic_labels(manifest, sheets)
        report = audit_report(manifest, gold, [], [{"name": "switch-minus-noop", "positive": {"condition": "SWITCH"}, "negative": {"condition": "NOOP"}}])
        self.assertTrue(report["measurement_sensitive"])
        self.assertEqual(report["contrasts"][0]["estimated_signed_measurement_bias"], 0)

    def test_census_audit_exact_and_error_detected(self):
        manifest, sheets = make_sample(population(), n=400, mode="audit")
        _, gold = synthetic_labels(manifest, sheets)
        contrasts = [{"name": "switch-minus-noop", "positive": {"condition": "SWITCH"}, "negative": {"condition": "NOOP"}}]
        self.assertFalse(audit_report(manifest, gold, [], contrasts)["measurement_sensitive"])
        for s in manifest["selected"]:
            if s["record"]["condition"] == "SWITCH" and s["record"]["scoring"]["given"]:
                gold[s["annotation_id"]]["human_given"] = 0
        self.assertTrue(audit_report(manifest, gold, [], contrasts)["measurement_sensitive"])


if __name__ == "__main__":
    unittest.main()
