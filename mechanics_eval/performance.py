"""Readable per-scenario and per-atomic-mechanic frequencies from saved verdicts.

Reporting only. Never infers new facts, changes statuses, resamples, or pools cohorts.
"""
import csv
from pathlib import Path

STATUSES = ("pass", "fail", "unknown")


def _scenario(row, ordinary):
    key = "ordinary_scenario_results_by_branch" if ordinary else "scenario_results_by_branch"
    branches = row.get(key)
    if branches:
        return branches[0]
    return row.get("scenario_result", row["outcome"])


def _stages(row, ordinary):
    key = "ordinary_measurements_by_branch" if ordinary else "measurements_by_branch"
    return (row.get(key) or row.get("measurements_by_branch") or [[], []])[0]


def _counts(verdicts):
    values = [v["status"] for v in verdicts]
    unknown_statuses = set(values) - set(STATUSES)
    if unknown_statuses:
        raise ValueError("Unrecognized scenario status: " + str(sorted(unknown_statuses)))
    n = len(values)
    p = values.count("pass")
    f = values.count("fail")
    u = values.count("unknown")
    return dict(attempts=n, passed=p, failed=f, unknown=u,
                pass_rate=p/n if n else None, upper_bound=(p+u)/n if n else None,
                decision_coverage=(p+f)/n if n else None)


def summarize(rows, ordinary=False):
    """Return separate scenario-type and atomic-stage frequencies for one cohort.

    Scenario pass rates always use attempted cases as the denominator; UNKNOWN
    cases are never counted as successes. Atomic stage rates use eligible checks
    and separately disclose prerequisite-blocked / inapplicable stages.
    """
    rows = list(rows)
    scenarios = []
    for name in sorted({r.get("scenario", "") for r in rows}):
        selected = [r for r in rows if r.get("scenario", "") == name]
        values = _counts([_scenario(r, ordinary) for r in selected])
        seeds = sorted({r["seed"] for r in selected if "seed" in r})
        samples = sorted({r.get("model_sample", 0) for r in selected})
        scenarios.append(dict(name=name, label=name.replace("_", " ").capitalize(),
            category=selected[0].get("category", "uncategorized"),
            scope=selected[0].get("scope", "static"),
            engine_seed_count=len(seeds), model_sample_count=len(samples),
            seeds=seeds, **values))

    stages = []
    keys = sorted({(s["name"], bool(s.get("required", False)))
                   for r in rows for s in _stages(r, ordinary)})
    for name, required in keys:
        selected = [s for r in rows for s in _stages(r, ordinary)
                    if s["name"] == name and bool(s.get("required", False)) == required]
        counts = {s: sum(v["status"] == s for v in selected)
                  for s in (*STATUSES, "not_tested", "not_applicable")}
        if sum(counts.values()) != len(selected):
            raise ValueError("Unrecognized atomic stage status")
        applicable = len(selected) - counts["not_applicable"]
        eligible = counts["pass"] + counts["fail"] + counts["unknown"]
        stages.append(dict(name=name, label=name.replace("_", " ").replace(".", ": "),
            category=selected[0].get("category", "uncategorized"),
            required=required, instances=len(selected), applicable=applicable,
            eligible=eligible, passed=counts["pass"], failed=counts["fail"],
            unknown=counts["unknown"], not_tested=counts["not_tested"],
            not_applicable=counts["not_applicable"],
            pass_rate=counts["pass"]/eligible if eligible else None,
            decision_coverage=(counts["pass"]+counts["fail"])/eligible if eligible else None,
            eligibility_coverage=eligible/applicable if applicable else None))
    return dict(scenarios=scenarios, atomic=stages)


def write_csv(path, performance):
    """Export a machine-readable table with explicit, non-pooled denominators."""
    path = Path(path)
    columns = ("kind", "name", "category", "required", "scope", "attempts",
               "eligible", "passed", "failed", "unknown", "not_tested",
               "not_applicable", "pass_rate", "decision_coverage",
               "eligibility_coverage", "engine_seed_count", "model_sample_count")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in performance["scenarios"]:
            writer.writerow({k: v for k, v in dict(kind="scenario", **item).items() if k in columns})
        for item in performance["atomic"]:
            writer.writerow({k: v for k, v in dict(kind="atomic", **item).items() if k in columns})
