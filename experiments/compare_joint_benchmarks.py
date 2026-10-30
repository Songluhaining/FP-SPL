"""按原本通过配置比较统一图方法与已保存的PULP/CLAP预测。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfp_spl.evaluation import _metrics


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def compare(root, baseline, system, clap_file=None):
    truth = read(baseline)
    clap = {(r['version'], r['configuration']): r for r in read(clap_file)} if clap_file else {}
    available = {}
    for file in sorted(root.glob('*/ranking.csv')):
        for row in read(file):
            available[(file.parent.name, row['configuration'])] = row
    completed = set()
    for version in {r['version'] for r in truth}:
        expected = {(r['version'], r.get('variant', r.get('configuration'))) for r in truth if r['version'] == version}
        if expected <= set(available):
            completed.add(version)
    rows = []
    for item in truth:
        key = item['version'], item.get('variant', item.get('configuration'))
        if item['version'] not in completed:
            continue
        result = available[key]
        is_bank = system == 'BankAccountTP'
        c = clap[key] if is_bank else item
        level = result['evidence_level']
        rows.append({'version': key[0], 'configuration': key[1], 'group': item['group'],
                     'label': int(item['label'] == 'FP') if is_bank else int(item['label']),
                     'pulp_prediction': int(item['pulp_prediction'] == 'FP') if is_bank else int(item['pulp_prediction']),
                     'pulp_score': float(item['pulp_score']),
                     'clap_prediction': int(c['prediction'] if is_bank else c['clap_prediction']),
                     'clap_score': float(c['score'] if is_bank else c['clap_score']),
                     'joint_prediction': int(level in {'confirmed_false_passing', 'strong_suspect'}),
                     'joint_score': {'insufficient': 0, 'weak_suspect': 1, 'strong_suspect': 2,
                                     'confirmed_false_passing': 3}[level] + float(result['score']),
                     'joint_evidence': level})
    labels = [r['label'] for r in rows]
    natural = {method: _metrics(labels, [r[method + '_prediction'] for r in rows],
                                [r[method + '_score'] for r in rows])
               for method in ('pulp', 'clap', 'joint')}
    budgets = {group: sum(r['pulp_prediction'] for r in rows if r['group'] == group)
               for group in {r['group'] for r in rows}}
    matched = {}
    for method in ('pulp', 'clap', 'joint'):
        selected = set()
        for group, budget in budgets.items():
            ranked = sorted((r for r in rows if r['group'] == group),
                            key=lambda r: (-r[method + '_score'], r['version'], r['configuration']))
            selected.update((r['version'], r['configuration']) for r in ranked[:budget])
        matched[method] = _metrics(labels, [(r['version'], r['configuration']) in selected for r in rows],
                                   [r[method + '_score'] for r in rows])
    return {'system': system, 'status': 'final' if len(rows) == len(truth) else 'interim_completed_versions_only',
            'expected_candidates': len(truth), 'evaluated_candidates': len(rows),
            'expected_versions': len({r['version'] for r in truth}), 'completed_versions': len(completed),
            'false_passing': sum(labels), 'true_passing': len(labels) - sum(labels),
            'evidence_levels': dict(Counter(r['joint_evidence'] for r in rows)),
            'group_budgets': budgets, 'natural_decisions': natural, 'same_group_budget': matched}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank-results', type=Path, required=True)
    parser.add_argument('--email-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    sources = [('bank', args.bank_results, Path('outputs/all73-comparison/matched_predictions.csv'),
                'BankAccountTP', Path('outputs/clap-fivefold-all73/predictions.csv')),
               ('email', args.email_results, Path('../email_trial/baselines/predictions-balanced.csv'),
                'Email', None)]
    for key, root, baseline, system, clap in sources:
        report, rows = compare(root, baseline, system, clap)
        reports[key] = report
        if rows:
            with (args.output / (key + '-predictions.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (args.output / 'comparison.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: {s: v[s] for s in ('status', 'completed_versions', 'evaluated_candidates')}
                      for k, v in reports.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
