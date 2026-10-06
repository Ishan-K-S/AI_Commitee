import csv
import json
from pathlib import Path
import tempfile
import unittest

from policy_checker.cli import main
from policy_checker.labeling import load_annotations
from test_checker import record


class WorkflowTests(unittest.TestCase):
    def test_complete_cli_workflow_and_failure_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.jsonl"
            raw.write_text("".join(json.dumps(record(
                "I cannot help." if i % 4 == 0 else "Final answer: 500" if i % 4 == 1 else "Consider 25 groups.",
                response_id=f"r{i}", condition="SWITCH" if i % 2 else "NOOP")) + "\n" for i in range(400)), encoding="utf-8")
            scored = root / "scored.jsonl"
            self.assertEqual(main(["score", str(raw), str(scored)]), 0)
            self.assertEqual(main(["summarize", str(scored), str(root / "summary.json")]), 0)
            self.assertEqual(json.loads((root / "summary.json").read_text())["overall"]["n"], 400)
            recognition_input = root / "recognition.jsonl"
            recognition_input.write_text(json.dumps({"response_id": "q1", "response": "B", "options": {"A": True, "B": False}, "question_negated": True, "expected_allowed": True}) + "\n")
            self.assertEqual(main(["recognition", str(recognition_input), str(root / "recognition-out.jsonl")]), 0)
            self.assertTrue(json.loads((root / "recognition-out.jsonl").read_text())["recognition"]["correct"])
            out = root / "validation"
            self.assertEqual(main(["sample", str(scored), str(out), "--seed", "12"]), 0)
            manifest_path = out / "private" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            lookup = {s["annotation_id"]: s for s in manifest["selected"]}
            paths = {name: out / f"annotator_{name}.csv" for name in ("A", "B")}
            with self.assertRaises(ValueError):
                load_annotations(manifest, paths)
            stored = {}
            for name, path in paths.items():
                with path.open(encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    fields, rows = reader.fieldnames, list(reader)
                for row in rows:
                    score = lookup[row["annotation_id"]]["record"]["scoring"]
                    row.update(human_given=str(int(score["given"])), human_refusal=str(int(score["refuse_proxy"])), human_help="1", human_correct="1")
                stored[name] = (fields, rows)
                self.write(path, fields, rows)
            self.assertEqual(main(["validate", str(manifest_path), str(paths["A"]), str(paths["B"]), str(root / "report.json")]), 0)
            self.assertTrue(json.loads((root / "report.json").read_text())["gate_pass"])
            contrasts_path = root / "contrasts.json"
            contrasts_path.write_text(json.dumps([{"name": "switch-noop", "positive": {"condition": "SWITCH"}, "negative": {"condition": "NOOP"}}]))
            self.assertEqual(main(["audit", str(manifest_path), str(paths["A"]), str(paths["B"]), str(root / "audit-report.json"), "--contrasts", str(contrasts_path)]), 0)
            self.assertTrue(json.loads((root / "audit-report.json").read_text())["measurement_sensitive"])
            fields, rows = stored["B"]
            shared = next(r for r in rows if len(lookup[r["annotation_id"]]["annotators"]) == 2)
            shared["human_given"] = "0" if shared["human_given"] == "1" else "1"
            self.write(paths["B"], fields, rows)
            _, _, unresolved = load_annotations(manifest, paths)
            self.assertEqual(unresolved, [shared["annotation_id"]])
            shared["response"] = "accidentally edited text"
            self.write(paths["B"], fields, rows)
            with self.assertRaises(ValueError):
                load_annotations(manifest, paths)

    @staticmethod
    def write(path, fields, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
