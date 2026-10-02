"""Weighted validation points and conservative finite-population audit bounds."""

from collections import defaultdict
import math


def divide(a, b):
    return a / b if b else None


def summarize(records):
    def rates(rows):
        scores = [r["scoring"] for r in rows]
        fields = ("given", "y", "refuse_proxy", "confirm_seeking", "bare_answer_proxy", "truncated")
        return {"n": len(rows), **{f: {"rate": divide(sum(s[f] for s in scores if s[f] is not None), sum(s[f] is not None for s in scores)),
                                      "observed_n": sum(s[f] is not None for s in scores)} for f in fields}}
    from .checker import CELL_FIELDS
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r[f] for f in CELL_FIELDS)].append(r)
    return {"overall": rates(records), "cells": [{"cell": dict(zip(CELL_FIELDS, key)), **rates(rows)} for key, rows in sorted(groups.items())],
            "note": "Descriptive unweighted response rates only. Not the study's model/channel-pooled confirmatory contrasts. REFUSE and bare-answer are proxies."}


def kappa(pairs):
    # Sampling weights correct validation oversampling; overlap selection is uniform.
    pairs = [(a, b, w) for a, b, w in pairs if a in ("0", "1") and b in ("0", "1")]
    total = sum(w for _, _, w in pairs)
    if not total:
        return None
    observed = sum(w for a, b, w in pairs if a == b) / total
    pa = sum(w for a, _, w in pairs if a == "1") / total
    pb = sum(w for _, b, w in pairs if b == "1") / total
    expected = pa * pb + (1 - pa) * (1 - pb)
    return divide(observed - expected, 1 - expected)


def validation_report(manifest, ratings, gold, unresolved):
    if unresolved:
        return {"status": "needs_adjudication", "unresolved_ids": unresolved, "gate_pass": False}

    def aggregate(samples):
        tp = tn = fp = fn = agree = total = 0.0
        for sample in samples:
            aid = sample["annotation_id"]
            w = sample["sampling_weight"]
            prediction = sample["record"]["scoring"]
            actual = gold[aid]
            pair = (bool(prediction["given"]), bool(actual["human_given"]))
            tp += w * (pair == (True, True))
            tn += w * (pair == (False, False))
            fp += w * (pair == (True, False))
            fn += w * (pair == (False, True))
            agree += w * (prediction["refuse_proxy"] == bool(actual["human_refusal"]))
            total += w
        return {"sample_n": len(samples), "weighted_n": total,
                "given_sensitivity": divide(tp, tp + fn), "given_specificity": divide(tn, tn + fp),
                "given_error_rate": divide(fp + fn, total), "refuse_proxy_agreement": divide(agree, total),
                "confusion_weighted": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}

    selected = manifest["selected"]
    overall = aggregate(selected)
    kappas = {}
    for field in ("human_given", "human_refusal"):
        pairs = [(ratings[s["annotation_id"]]["A"][field], ratings[s["annotation_id"]]["B"][field], s["sampling_weight"])
                 for s in selected if len(s["annotators"]) == 2]
        kappas[field] = kappa(pairs)
    checks = {"given_sensitivity": overall["given_sensitivity"] is not None and overall["given_sensitivity"] >= .90,
              "given_specificity": overall["given_specificity"] is not None and overall["given_specificity"] >= .90,
              "refuse_proxy_agreement": overall["refuse_proxy_agreement"] is not None and overall["refuse_proxy_agreement"] >= .85,
              "given_kappa": kappas["human_given"] is not None and kappas["human_given"] >= .8,
              "refusal_kappa": kappas["human_refusal"] is not None and kappas["human_refusal"] >= .8}
    uncertain_overlap = any(r[f] == "uncertain" for aid, raters in ratings.items() if len(raters) == 2 for r in raters.values() for f in ("human_given", "human_refusal"))
    checks["complete_overlap_binary_ratings"] = not uncertain_overlap
    splits = {}
    for field in ("condition", "direction", "remedy", "model_id"):
        splits[field] = {v: aggregate([s for s in selected if s["record"][field] == v]) for v in sorted({s["record"][field] for s in selected})}
    return {"status": "complete", "overall": overall, "kappa_pre_adjudication": kappas,
            "checks": checks, "gate_pass": all(checks.values()), "by": splits,
            "interpretation": "Point-estimate development gate only. Not confidence-certified accuracy or proof of no differential measurement error.",
            "weights": "Inverse inclusion probabilities; overall results target the supplied response population."}


def audit_report(manifest, gold, unresolved, contrasts, threshold=.02, alpha=.05):
    """Hoeffding upper bounds for fixed-population mismatch rates, union over strata.

    These deliberately conservative bounds include unsigned errors and do not assume
    matching error rates in the two arms. Census strata have exact error rates.
    """
    if unresolved:
        return {"status": "needs_adjudication", "measurement_sensitive": None, "unresolved_ids": unresolved}
    if not contrasts or not 0 < alpha < 1 or not 0 < threshold < 1:
        raise ValueError("provide contrasts and valid alpha/threshold")
    samples_by_stratum = defaultdict(list)
    for s in manifest["selected"]:
        samples_by_stratum[s["stratum"]].append(s)
    bounds = {}
    for sid, spec in manifest["strata"].items():
        samples = samples_by_stratum[sid]
        errors = sum(s["record"]["scoring"]["given"] != bool(gold[s["annotation_id"]]["human_given"]) for s in samples)
        rate = errors / len(samples)
        upper = rate if len(samples) == spec["N"] else min(1.0, rate + math.sqrt(math.log(len(manifest["strata"]) / alpha) / (2 * len(samples))))
        bounds[sid] = {"errors": errors, "n": len(samples), "N": spec["N"], "error_upper": upper}
    results = []
    population = manifest["population"]

    def matches(row, filters):
        if not filters or any(key not in row for key in filters):
            raise ValueError("contrast filters must be nonempty and use population fields")
        return all(row[key] == value for key, value in filters.items())

    for contrast in contrasts:
        arms = [[r for r in population if matches(r, contrast[arm])] for arm in ("positive", "negative")]
        if not all(arms):
            raise ValueError(f"{contrast['name']}: empty contrast arm")
        if {r["response_id"] for r in arms[0]} & {r["response_id"] for r in arms[1]}:
            raise ValueError("contrast arms overlap")
        coefficients = {}
        for sign, arm in zip((1, -1), arms):
            total = sum(r["analysis_weight"] for r in arm)
            for r in arm:
                coefficients[r["response_id"]] = sign * r["analysis_weight"] / total
        point_bias = 0.0
        for s in manifest["selected"]:
            r = s["record"]
            human_y = gold[s["annotation_id"]]["human_given"] if r["direction"] == "tighten" else 1 - gold[s["annotation_id"]]["human_given"]
            point_bias += coefficients.get(r["response_id"], 0) * s["sampling_weight"] * (r["scoring"]["y"] - human_y)
        upper_bias = 0.0
        for sid, spec in manifest["strata"].items():
            values = [abs(coefficients.get(r["response_id"], 0)) for r in population if r["stratum"] == sid]
            # Bound the worst placement of misclassifications when coefficients vary.
            upper_bias += min(sum(values), spec["N"] * max(values, default=0) * bounds[sid]["error_upper"])
        relevant_strata = {r["stratum"] for r in population if r["response_id"] in coefficients}
        if all(manifest["strata"][sid]["n"] == manifest["strata"][sid]["N"] for sid in relevant_strata):
            upper_bias = abs(point_bias)
        auto = sum(coefficients.get(r["response_id"], 0) * r["y"] for r in population)
        results.append({"name": contrast["name"], "automated_contrast": auto,
                        "estimated_signed_measurement_bias": point_bias,
                        "estimated_human_contrast": auto - point_bias,
                        "absolute_bias_upper_bound": upper_bias,
                        "measurement_sensitive": upper_bias >= threshold,
                        "reason": "estimated_bias_exceeds_threshold" if abs(point_bias) >= threshold else "cannot_rule_out_threshold" if upper_bias >= threshold else "bounded_below_threshold"})
    return {"status": "complete", "threshold": threshold, "confidence": 1 - alpha,
            "measurement_sensitive": any(r["measurement_sensitive"] for r in results),
            "contrasts": results, "stratum_error_bounds": bounds,
            "method": "Simultaneous finite-population Hoeffding error bounds across sampling strata; conservative and conditional on accurate human labels. A flag is uncertainty or estimated distortion, not proof of harm."}
