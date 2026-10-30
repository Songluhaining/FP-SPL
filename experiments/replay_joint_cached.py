"""离线回放已有轨迹以验证联合构图；不是大模型补全后的正式实验。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfp_spl.brfl_runtime import BrflRuntime, Execution
from cfp_spl.configurations import load_configurations
from cfp_spl.domain import JavaTestCase, TestPair
from cfp_spl.factor_graph import load_factor_graph
from cfp_spl.feature_sources import load_feature_map, source_identity
from cfp_spl.joint_causal import assess_joint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--available', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--java-home', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    runtime = BrflRuntime(args.output / 'runtime', args.java_home)
    runtime.build_brfl()
    groups = defaultdict(list)
    for item in json.loads(args.available.read_text(encoding='utf-8')):
        groups[(item['version'], item['row']['passing_configuration'])].append(item)
    reports = []
    for index, ((version, name), items) in enumerate(groups.items()):
        if args.limit and index >= args.limit:
            break
        configs = {c.name: c for c in load_configurations(args.dataset / version)}
        output = args.output / version / name
        output.mkdir(parents=True, exist_ok=True)
        try:
            executions, seen, pairs = [], set(), []
            expected_names, local_statement_count = set(), 0
            for item in items:
                row = item['row']
                sides = []
                for side, stage_str in zip(('passing', 'failing'), item['stages']):
                    stage = Path(stage_str)
                    config = configs[row[side + '_configuration']]
                    cls, method = row[side + '_test'].rsplit('.', 1)
                    test = JavaTestCase(cls, method, Path('cached'), '', '')
                    execution = Execution('passed' if side == 'passing' else 'failed',
                                          stage / 'factor-graph.tsv', stage / 'test-run.log', None, None)
                    structure = load_factor_graph(execution.graph_path)
                    sides.append((config, test, execution, structure))
                    key = str(stage)
                    if key not in seen:
                        executions.append((config, test, execution))
                        seen.add(key)
                        fmap = load_feature_map(config.variant_dir)
                        expected_names.update(source_identity(s.stable_key, fmap) for s in structure.statements.values()
                                              if source_identity(s.stable_key, fmap) != s.stable_key)
                        local_statement_count += len(structure.statements)
                pairs.append((row, sides))
            graph = load_factor_graph(runtime.infer_joint(executions, output))
            identities = [s.stable_key for s in graph.statements.values()]
            mapped = {key for key in identities if not key.startswith('unmapped:')}
            observations = [v['observed_value'] for v in graph.values.values() if v['observed']]
            assert len(identities) == len(set(identities)), 'duplicate statement identity'
            assert mapped == expected_names, 'source mapping mismatch'
            assert True in observations and False in observations, 'missing mixed outcomes'
            assert len(graph.values) == sum(len(load_factor_graph(e.graph_path).values) for _, _, e in executions), 'dynamic value nodes merged'
            assessments = []
            for row, (left, right) in pairs:
                a, ta, ea, ga = left
                b, tb, eb, gb = right
                pair = TestPair(tb, ta, row['test_provenance'], row['test_similarity'], row['oracle_similarity'])
                assessments.append(assess_joint(a.name, b.name, pair, graph, ga, gb,
                                               load_feature_map(a.variant_dir), load_feature_map(b.variant_dir)))
            best = max(assessments, key=lambda a: a.score)
            report = {'version': version, 'configuration': name, 'status': 'validated',
                      'executions': len(executions), 'local_statement_total': local_statement_count,
                      'joint_statements': len(identities), 'mapped_statements': len(mapped),
                      'dynamic_values': len(graph.values), 'factors': len(graph.factors),
                      'passed_observations': observations.count(True), 'failed_observations': observations.count(False),
                      'evidence_level': best.evidence_level, 'score': best.score}
            (output / 'pair-assessments.json').write_text(json.dumps([asdict(a) for a in assessments], ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as exc:
            report = {'version': version, 'configuration': name, 'status': 'error', 'error': str(exc)}
        reports.append(report)
        (args.output / 'replay-summary.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
