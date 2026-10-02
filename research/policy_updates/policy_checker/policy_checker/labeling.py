"""Finite-population stratified sampling and separate public/private artifacts."""

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random

from .checker import CELL_FIELDS

LABEL_FIELDS = ("human_given", "human_refusal", "human_help", "human_correct", "notes")
SHEET_FIELDS = ("annotation_id", "problem", "response", "reference_answer", "reference_intermediates") + LABEL_FIELDS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path):
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{n}: expected object")
        rows.append(row)
    ids = [r["response_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("response_id must be unique; select one attempt per planned response before sampling")
    return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def safe_cell(value):
    value = str(value)
    # CSV quoting alone does not prevent spreadsheet formula execution.
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def write_sheet(path, rows, fields=SHEET_FIELDS):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows({key: safe_cell(row.get(key, "")) for key in fields} for row in rows)


def make_sample(records, n=300, seed=1, mode="validation", refuse_min=60, overlap_fraction=.25):
    if mode not in ("validation", "audit"):
        raise ValueError("unknown sample mode")
    if not 0 <= overlap_fraction <= 1:
        raise ValueError("overlap_fraction must be in [0, 1]")
    if not 0 < n <= len(records):
        raise ValueError("sample size must be positive and no greater than the population")
    records = sorted(records, key=lambda r: r["response_id"])
    if len({r["response_id"] for r in records}) != len(records):
        raise ValueError("duplicate response IDs")
    if len({r["scoring"]["rules_sha256"] for r in records}) != 1:
        raise ValueError("population mixes checker versions")
    if len({r["scoring"]["checker_code_sha256"] for r in records}) != 1:
        raise ValueError("population mixes checker implementations")
    groups = defaultdict(list)
    for r in records:
        key = tuple(r[f] for f in CELL_FIELDS)
        if mode == "validation":
            key += (str(r["scoring"]["refuse_proxy"]),)
        groups[key].append(r)
    keys = sorted(groups)
    if n < len(keys):
        raise ValueError(f"{len(keys)} nonempty strata require at least that many sampled responses; do not silently drop cells")
    if mode == "audit":
        refuse_min = 0
    if refuse_min < 0 or refuse_min > n:
        raise ValueError("invalid refusal quota")
    refusal_keys = [k for k in keys if mode == "validation" and k[-1] == "True"]
    if sum(len(groups[k]) for k in refusal_keys) < refuse_min:
        raise ValueError("population has fewer rule-REFUSE responses than the requested quota")
    allocations = {k: 1 for k in keys}

    def allocate(candidates):
        choices = [k for k in candidates if allocations[k] < len(groups[k])]
        if not choices:
            raise ValueError("cannot meet quota while preserving nonzero inclusion in every stratum")
        key = max(choices, key=lambda k: (n * len(groups[k]) / len(records) - allocations[k], k))
        allocations[key] += 1

    while sum(allocations[k] for k in refusal_keys) < refuse_min:
        if sum(allocations.values()) >= n:
            raise ValueError("sample too small for both quota and complete cell coverage")
        allocate(refusal_keys)
    while sum(allocations.values()) < n:
        allocate(keys)
    rng = random.Random(seed)
    selected, strata, population = [], {}, []
    for idx, key in enumerate(keys):
        sid = f"s{idx:04d}"
        group = groups[key]
        nh, nh_sample = len(group), allocations[key]
        strata[sid] = {"key": list(key), "N": nh, "n": nh_sample}
        for r in group:
            population.append({"response_id": r["response_id"], "stratum": sid,
                               **{f: r[f] for f in CELL_FIELDS},
                               "analysis_weight": r.get("analysis_weight", 1.0),
                               "given": r["scoring"]["given"], "y": r["scoring"]["y"]})
        for r in rng.sample(group, nh_sample):
            aid = f"a{rng.getrandbits(128):032x}"
            selected.append({"annotation_id": aid, "response_id": r["response_id"],
                             "stratum": sid, "inclusion_probability": nh_sample / nh,
                             "sampling_weight": nh / nh_sample, "record": r})
    rng.shuffle(selected)
    overlap_n = math.floor(n * overlap_fraction + .5)
    overlap = set(rng.sample([r["annotation_id"] for r in selected], overlap_n))
    sheets = {"A": [], "B": []}
    solo = [r["annotation_id"] for r in selected if r["annotation_id"] not in overlap]
    rng.shuffle(solo)
    solo_a = set(solo[::2])
    for s in selected:
        r = s["record"]
        row = {"annotation_id": s["annotation_id"], "problem": r["problem"], "response": r["response"],
               "reference_answer": str(r["final_answer"]),
               "reference_intermediates": json.dumps(r["intermediate_values"], ensure_ascii=False),
               **{f: "" for f in LABEL_FIELDS}}
        assigned = ["A", "B"] if s["annotation_id"] in overlap else ["A" if s["annotation_id"] in solo_a else "B"]
        s["annotators"] = assigned
        for name in assigned:
            sheets[name].append(row.copy())
    for sheet in sheets.values():
        rng.shuffle(sheet)
    manifest = {"schema_version": 1, "mode": mode, "seed": seed, "sample_size": n,
                "population_size": len(records), "population_sha256": digest(records),
                "rules_sha256": records[0]["scoring"]["rules_sha256"],
                "checker_code_sha256": records[0]["scoring"]["checker_code_sha256"],
                "overlap_count": overlap_n, "refuse_quota": refuse_min, "strata": strata,
                "selected": selected, "population": population,
                "warning": "PRIVATE: contains conditions, checker predictions and sampling weights. Do not send to annotators."}
    return manifest, sheets


def export_sample(directory, manifest, sheets):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    private = directory / "private"
    private.mkdir()
    write_json(private / "manifest.json", manifest)
    for name, sheet in sheets.items():
        write_sheet(directory / f"annotator_{name}.csv", sheet)
    (directory / "SHARING.txt").write_text(
        "Send each annotator ONLY their CSV and the labeling guide. Do not share private/.\n"
        "CSV text starting with a spreadsheet formula character has a leading apostrophe for safety.\n"
        "Only fill human_* and notes columns; retain annotation_id unchanged.\n", encoding="utf-8")


def load_annotations(manifest, paths, adjudication=None):
    expected = {r["annotation_id"]: r for r in manifest["selected"]}
    ratings = defaultdict(dict)
    for name, path in paths.items():
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                aid = row["annotation_id"]
                if aid not in expected or name not in expected[aid]["annotators"]:
                    raise ValueError("unknown or incorrectly assigned annotation_id")
                if name in ratings[aid]:
                    raise ValueError("duplicate annotation")
                source = expected[aid]["record"]
                readonly = {"problem": source["problem"], "response": source["response"],
                            "reference_answer": str(source["final_answer"]),
                            "reference_intermediates": json.dumps(source["intermediate_values"], ensure_ascii=False)}
                if any(row.get(f) != safe_cell(value) for f, value in readonly.items()):
                    raise ValueError(f"{aid}: read-only annotation text changed; restore the original sheet")
                for field in LABEL_FIELDS[:-1]:
                    if row.get(field, "") not in ("0", "1", "uncertain"):
                        raise ValueError(f"{aid}: {field} must be 0, 1, or uncertain; incomplete labels cannot pass")
                ratings[aid][name] = row
    resolutions = {}
    if adjudication:
        with Path(adjudication).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                aid = row["annotation_id"]
                if aid not in expected or aid in resolutions:
                    raise ValueError("invalid adjudication ID")
                resolutions[aid] = row
    gold, unresolved = {}, []
    for aid, sample in expected.items():
        rows = ratings.get(aid, {})
        if set(rows) != set(sample["annotators"]):
            raise ValueError(f"{aid}: missing assigned annotation")
        disagreement = any(len({r[f] for r in rows.values()}) > 1 or any(r[f] == "uncertain" for r in rows.values()) for f in LABEL_FIELDS[:-1])
        if disagreement and aid not in resolutions:
            unresolved.append(aid)
            continue
        chosen = resolutions.get(aid, next(iter(rows.values())))
        if any(chosen.get(f) not in ("0", "1") for f in LABEL_FIELDS[:-1]):
            raise ValueError(f"{aid}: adjudicated labels must be 0 or 1")
        gold[aid] = {f: int(chosen[f]) for f in LABEL_FIELDS[:-1]}
    return ratings, gold, unresolved
