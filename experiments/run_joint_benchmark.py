"""按缺陷版本并发运行统一图实验，保存可恢复的版本级进度。"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfp_spl.pipeline import CounterfactualPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--java-home', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    def pipeline():
        return CounterfactualPipeline(args.dataset, args.output, args.java_home,
                                      deepseek_key_file=args.key_file)
    pipeline().runtime.build_brfl()
    versions = sorted(p for p in args.dataset.iterdir() if (p / 'config.report.csv.done').is_file())
    summary = {}
    summary_path = args.output / 'benchmark-summary.json'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
    def run(version):
        runner = pipeline()
        # 每个版本保留独立源码摘要和参数清单，避免多线程覆盖。
        runner_manifest = args.output / version.name / 'run-manifest.json'
        runner_manifest.parent.mkdir(parents=True, exist_ok=True)
        sources = [*Path('cfp_spl').glob('*.py'), *Path('java').rglob('*.java'),
                   *Path('vendor/brfl/src').rglob('*.java')]
        import hashlib
        runner_manifest.write_text(json.dumps({
            'started_utc': datetime.now(timezone.utc).isoformat(), 'dataset': str(args.dataset),
            'version': version.name, 'model': runner.generator.deepseek_model,
            'neighbors_with_ties': 1, 'max_generation_rounds': 3, 'ground_truth_used': False,
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        try:
            return {'status': 'complete', **runner._run_version(version, False)}
        except Exception as exc:
            return {'status': 'error', 'error': str(exc), 'traceback': traceback.format_exc()}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run, version): version for version in versions
                if summary.get(version.name, {}).get('status') != 'complete'}
        for future in as_completed(jobs):
            version = jobs[future].name
            summary[version] = future.result()
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'version': version,
                              **summary[version]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
