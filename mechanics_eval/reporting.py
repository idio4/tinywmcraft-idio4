"""Local machine-readable reports, with provenance, missingness and counts."""
import csv
import json
from pathlib import Path

from .metrics import SCALARS


def write_report(out, report):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    fields = ['scenario', 'category', 'valid_attempt', 'seed', 'model_sample', 'horizon', 'n_steps',
              'terminal_censored', 'milestone_observed', *SCALARS]
    with (out / 'samples.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(report['samples'])
    with (out / 'summary.csv').open('w', newline='') as file:
        fields = ['group', 'samples', 'steps'] + [f'{m}_{s}' for m in SCALARS for s in ('mean', 'n', 'total', 'se')]
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for name, group in report['aggregate']['groups'].items():
            writer.writerow({'group': name, 'samples': group['samples'], 'steps': group['steps'],
                **{f'{m}_{s}': group[m][s] for m in SCALARS for s in ('mean', 'n', 'total', 'se')}})
    text = ['# Level 1 mechanics evaluation', '', f"Mode: {report['mode']}. Status: {report['status']}.",
            f"Samples: {len(report['samples'])}; failures: {len(report['errors'])}.", '',
            'Nulls mean unavailable; reference-only runs contain no learned-model scores.',
            'Reward and termination heads do not establish that an event appeared in generated RGB.', '',
            '| Category / horizon / validity | Samples | Visual MSE | Outcome accuracy (n) | Persistence (n) |',
            '|---|---:|---:|---:|---:|']
    def fmt(value):
        return 'unavailable' if value is None else f'{value:.4g}'
    for name, group in report['aggregate']['groups'].items():
        outcome, persistence = group['outcome_accuracy'], group['persistence_success']
        text.append(f"| {name} | {group['samples']} | {fmt(group['visual_mse']['mean'])} | "
                    f"{fmt(outcome['mean'])} ({outcome['n']}) | {fmt(persistence['mean'])} ({persistence['n']}) |")
    text += ['', 'See results.json for extraction validation, event counts, coverage, counterfactuals and provenance.',
             'Uncertainty: reported SE is descriptive across seed/model samples, not a generalization guarantee.',
             'Model repeats share reference trajectories and are not independent environment samples.']
    if report['errors']:
        text += ['', '## Errors', ''] + [f"- {item}" for item in report['errors']]
    (out / 'summary.md').write_text('\n'.join(text) + '\n')
