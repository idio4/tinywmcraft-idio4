"""Readable per-scenario and per-atomic-mechanic frequencies from saved verdicts.

Reporting only. Never infers new facts, changes statuses, resamples, or pools cohorts.
"""
import csv
from pathlib import Path

STATUSES = ("pass", "fail", "unknown")


def scenario_verdict(row, ordinary=False):
    key = "ordinary_scenario_results_by_branch" if ordinary else "scenario_results_by_branch"
    branches = row.get(key) or row.get("scenario_results_by_branch")
    if branches:
        return branches[0]
    if "scenario_result" in row:
        return row["scenario_result"]
    if row.get("branch_outcomes"):
        return row["branch_outcomes"][0]
    return row["outcome"]


def atomic_stages(row, ordinary=False):
    key = "ordinary_measurements_by_branch" if ordinary else "measurements_by_branch"
    return row.get(key, row.get("measurements_by_branch", [[], []]))[0]


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
    for name, scope in sorted({(r.get("scenario", ""), r.get("scope", "static")) for r in rows}):
        selected = [r for r in rows if r.get("scenario", "") == name and r.get("scope", "static") == scope]
        values = _counts([scenario_verdict(r, ordinary) for r in selected])
        seeds = sorted({r["seed"] for r in selected if "seed" in r})
        samples = sorted({r.get("model_sample", 0) for r in selected})
        counts_per_seed = [sum(r.get("seed") == seed for r in selected) for seed in seeds]
        scenarios.append(dict(name=name, scenario=name, label=name.replace("_", " ").capitalize(),
            category=selected[0].get("category", "uncategorized"),
            scope=scope,
            engine_seed_count=len(seeds), model_sample_count=len(samples),
            samples_per_seed_min=min(counts_per_seed) if counts_per_seed else None,
            samples_per_seed_max=max(counts_per_seed) if counts_per_seed else None,
            seeds=seeds, **values))

    stages = []
    keys = sorted({(r.get("scenario", ""), s["name"], s.get("category", "uncategorized"), bool(s.get("required", False)), r.get("scope", "static"))
                   for r in rows for s in atomic_stages(r, ordinary)})
    for scenario, name, category, required, scope in keys:
        selected = [s for r in rows if r.get("scope", "static") == scope and r.get("scenario", "") == scenario for s in atomic_stages(r, ordinary)
                    if s["name"] == name and s.get("category", "uncategorized") == category and bool(s.get("required", False)) == required]
        counts = {s: sum(v["status"] == s for v in selected)
                  for s in (*STATUSES, "not_tested", "not_applicable")}
        if sum(counts.values()) != len(selected):
            raise ValueError("Unrecognized atomic stage status")
        applicable = len(selected) - counts["not_applicable"]
        eligible = counts["pass"] + counts["fail"] + counts["unknown"]
        stages.append(dict(name=name, label=name.replace("_", " ").replace(".", ": "),
            scenario=scenario, category=category, scope=scope,
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
    columns = ("kind", "name", "scenario", "category", "required", "scope", "attempts",
               "eligible", "passed", "failed", "unknown", "not_tested",
               "not_applicable", "pass_rate", "decision_coverage",
               "eligibility_coverage", "engine_seed_count", "model_sample_count",
               "samples_per_seed_min", "samples_per_seed_max", "instances", "applicable")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in performance["scenarios"]:
            writer.writerow({k: v for k, v in dict(kind="scenario", **item).items() if k in columns})
        for item in performance["atomic"]:
            writer.writerow({k: v for k, v in dict(kind="atomic", **item).items() if k in columns})


def markdown(performance):
    """Use the same counts and denominators as JSON, CSV and the viewer."""
    def pct(value):
        return f'{value:.1%}' if value is not None else 'N/A'
    lines = ['', '## Scenario pass frequency by mechanic', '',
             'PASS / attempted trials; UNKNOWN remains in the denominator. Frequencies describe these controlled trials, not general competence.', '',
             '| Scenario | Scope | Pass / attempts | Pass rate | Fail | Unknown | Decision coverage | Engine seeds | Draws per seed (min–max) |',
             '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for row in performance['scenarios']:
        draws = f"{row['samples_per_seed_min']}–{row['samples_per_seed_max']}" if row['samples_per_seed_min'] is not None else 'N/A'
        lines.append(f"| {row['name']} | {row['scope']} | {row['passed']}/{row['attempts']} | {pct(row['pass_rate'])} | {row['failed']} | {row['unknown']} | {pct(row['decision_coverage'])} | {row['engine_seed_count']} | {draws} |")
    for required, title in ((True, 'Required atomic checks'), (False, 'Separate diagnostic checks')):
        lines += ['', '## '+title, '',
                  'PASS / eligible checks. Eligibility is eligible / applicable; NOT_TESTED is a prerequisite gap and NOT_APPLICABLE is excluded. Diagnostics do not determine scenario verdicts.', '',
                  '| Scenario | Atomic mechanic | Scope | Pass / eligible | Pass rate | Fail | Unknown | NOT_TESTED | N/A | Eligibility coverage |',
                  '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in performance['atomic']:
            if row['required'] == required:
                lines.append(f"| {row['scenario']} | {row['name']} | {row['scope']} | {row['passed']}/{row['eligible']} | {pct(row['pass_rate'])} | {row['failed']} | {row['unknown']} | {row['not_tested']} | {row['not_applicable']} | {pct(row['eligibility_coverage'])} |")
    return lines
