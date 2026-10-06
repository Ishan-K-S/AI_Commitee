import argparse
import json
from pathlib import Path
import sys

from .checker import Checker, parse_recognition
from .labeling import export_sample, load_annotations, make_sample, read_jsonl, write_json
from .metrics import audit_report, validation_report, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline rule checker and blinded human-validation tools")
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="Score response records in JSONL; preserve raw fields")
    score.add_argument("input")
    score.add_argument("output")
    score.add_argument("--rules", help="Alternative versioned rules file")
    summary = sub.add_parser("summarize", help="Descriptive rates overall and by experimental cell")
    summary.add_argument("input")
    summary.add_argument("output")
    sample = sub.add_parser("sample", help="Create sheets and a PRIVATE sampling manifest")
    sample.add_argument("input")
    sample.add_argument("output_directory")
    sample.add_argument("--mode", choices=("validation", "audit"), default="validation")
    sample.add_argument("--n", type=int, help="Default 300 validation, 100 audit")
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--refuse-min", type=int, default=60)
    sample.add_argument("--overlap", type=float, default=.25)
    for name in ("validate", "audit"):
        command = sub.add_parser(name)
        command.add_argument("manifest")
        command.add_argument("annotator_a")
        command.add_argument("annotator_b")
        command.add_argument("output")
        command.add_argument("--adjudication")
        if name == "audit":
            command.add_argument("--contrasts", required=True, help="JSON list of named positive/negative population filters")
            command.add_argument("--threshold", type=float, default=.02)
    recognition = sub.add_parser("recognition", help="Parse JSONL recognition records with explicit polarity metadata")
    recognition.add_argument("input")
    recognition.add_argument("output")
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "output") and Path(args.output).exists():
            raise ValueError("output already exists; use a fresh run path to preserve evidence")
        if args.command == "score":
            checker = Checker(args.rules) if args.rules else Checker()
            rows = [checker.score(row) for row in read_jsonl(args.input)]
            Path(args.output).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
        elif args.command == "sample":
            n = args.n if args.n is not None else 300 if args.mode == "validation" else 100
            manifest, sheets = make_sample(read_jsonl(args.input), n, args.seed, args.mode, args.refuse_min, args.overlap)
            export_sample(args.output_directory, manifest, sheets)
        elif args.command == "summarize":
            write_json(args.output, summarize(read_jsonl(args.input)))
        elif args.command == "recognition":
            rows = [{**r, "recognition": parse_recognition(r["response"], r["options"], r["question_negated"], r["expected_allowed"])} for r in read_jsonl(args.input)]
            Path(args.output).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        else:
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            ratings, gold, unresolved = load_annotations(manifest, {"A": args.annotator_a, "B": args.annotator_b}, args.adjudication)
            if args.command == "validate":
                report = validation_report(manifest, ratings, gold, unresolved)
            else:
                contrasts = json.loads(Path(args.contrasts).read_text(encoding="utf-8"))
                report = audit_report(manifest, gold, unresolved, contrasts, args.threshold)
            write_json(args.output, report)
        return 0
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
