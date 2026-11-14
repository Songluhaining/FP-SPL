"""三系统完整多缺陷实验：案例独立推断，比较方法按系统及缺陷层五折。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .inventory import records, atomic_json
from .joint_fault import JointEvidence, infer_joint
from .joint_inputs import load_evidence, verify_unchanged

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'experimental_results/multifault_complete'
SCENARIO = 'multifault_joint'
PARAMETERS = dict(particles=256, points=9, sparsity=1.5, sweeps=2)
SEEDS = (42, 2026)
KEYS = ['system', 'version', 'configuration']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k:None if isinstance(v,(float,np.floating)) and np.isnan(v) else v for k,v in row.items()} for row in rows)
    os.replace(temporary, path)


def folder(job):
    return Path(job['path']) / '.fp_research' / SCENARIO


def code_hash(stage):
    names = ['joint_fault.py', 'joint_inputs.py', 'latent_fault.py', 'inventory.py'] if stage == 'joint' else [
        'case_study.py', 'baseline_features.py', 'baseline_cache.py', 'slice_runner.py', 'slice_reuse.py',
        'checks/SliceProfile.java', 'multifault_cache.py']
    return {n: digest(ROOT / 'fp_research' / n) for n in names}


def manifest():
    location = read(ROOT / 'fp_research/dataset_locations.json')
    jobs = []
    for system in location['active_multifault']:
        base = Path(location['systems'][system])
        for stratum in (2, 3):
            root = base / f'4wise-{base.name}-{stratum}BUG-Full'
            versions = sorted(root.glob('_MultipleBugs_*'))
            if not versions:
                raise ValueError(f'缺少多缺陷案例目录 {root}')
            for v in versions:
                with (v / 'config.report.csv.done').open(encoding='utf-8-sig', newline='') as stream:
                    rows = list(csv.reader(stream))[1:]
                states = [r[-1].strip() for r in rows]
                if set(states) - {'__PASSED__', '__FAILED__'}:
                    raise ValueError(f'未知原始状态 {v}')
                jobs.append(dict(system=system, stratum=stratum, version=v.name, path=str(v),
                                 configurations=len(rows), candidates=states.count('__PASSED__'),
                                 report_sha256=digest(v / 'config.report.csv.done')))
    spec = dict(jobs=jobs, parameters=PARAMETERS, seeds=list(SEEDS), threshold=.5,
                joint_protocol='每个案例单独推断；不输入真实故障数量或评价标签；两种种子平均',
                stability='最大候选分差>0.05时依次增加到1024、4096粒子；仍不稳则明确标记',
                baseline_protocol='各系统的双缺陷与三缺陷分别按版本固定五折；仅训练折标签供CLAP使用',
                primary_label='变异位置存在于启用特征的产品中；另报实际执行次数为正的标签',
                deferred=location['deferred_multifault'])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / 'manifest.json'
    if target.exists():
        old=read(target)
        if old['jobs'] != jobs:
            raise ValueError('输入清单改变，请先核对已有实验，不能静默覆盖')
        if old==spec and (OUTPUT/'inventory.csv').exists():return jobs
    atomic_json(target, spec)
    write_csv(OUTPUT / 'inventory.csv', jobs)
    return jobs


def fit(job):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    started = time.monotonic()
    v = Path(job['path']); out = folder(job)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'joint.json'
    if marker.exists():
        old = read(marker)
        if old['code'] == code_hash('joint') and old['parameters'] == PARAMETERS and old.get('numerical_threads')==1:
            verify_unchanged(v, read(out / 'evidence.json'))
            for name, sha in old['artifacts'].items():
                if digest(out / name) != sha: raise ValueError('联合缓存摘要不符')
            return 'cached'
    e, info = load_evidence(v)
    atomic_json(out / 'inventory.json', list(records(v)))
    atomic_json(out / 'evidence.json', info)
    np.savez_compressed(out / 'evidence.npz', **{k: getattr(e, k) for k in e.__dataclass_fields__})
    selected = np.array(info['passing'], dtype=bool)
    attempts = []
    for particles, sweeps in [(256, 2), (1024, 3), (4096, 4)]:
        runs = [infer_joint(e, seed=seed, **dict(PARAMETERS, particles=particles, sweeps=sweeps)) for seed in SEEDS]
        difference = float(np.max(np.abs(runs[0]['score'][selected] - runs[1]['score'][selected]))) if selected.any() else 0.
        attempts.append(dict(particles=particles, sweeps=sweeps, difference=difference))
        if difference <= .05: break
    means = {k: np.mean([r[k] for r in runs], axis=0) for k in ('score', 'lower', 'upper')}
    predictions = [dict(system=job['system'], version=v.name, configuration=name, stratum=job['stratum'],
                        method='joint', score=float(means['score'][i]), prediction=int(means['score'][i] >= .5),
                        lower=float(means['lower'][i]), upper=float(means['upper'][i]), status='scored',
                        stable=int(difference <= .05)) for i, name in enumerate(info['configurations']) if selected[i]]
    if len(predictions) != job['candidates']: raise ValueError('联合预测遗漏候选')
    write_csv(out / 'joint.csv', predictions)
    np.savez_compressed(out / 'posterior-states.npz', **{f'seed{s}': r['states'] for s, r in zip(SEEDS, runs)})
    verify_unchanged(v, info)
    atomic_json(marker, dict(code=code_hash('joint'), parameters=PARAMETERS, attempts=attempts,numerical_threads=1,
        stable=difference <= .05, seconds=time.monotonic()-started, tests=info['tests'], classes=len(info['classes']),
        patterns=info['patterns'], runs=[{k: r[k] for k in ('seed', 'particles', 'trace', 'epsilon', 'log_evidence')} for r in runs],
        artifacts={n: digest(out / n) for n in ('evidence.json', 'evidence.npz', 'joint.csv', 'posterior-states.npz')}))
    return f"verified classes={len(info['classes'])} seconds={time.monotonic()-started:.1f} difference={difference:.4f}"


def prepare(job):
    from .case_study import raw_digest, slices, features
    from .compare import load_features
    v = Path(job['path']); out = folder(job)
    out.mkdir(parents=True, exist_ok=True)
    before = raw_digest(v)
    stamp = out / 'baselines.json'
    if stamp.exists():
        old = read(stamp)
        if old['raw_digest'] == before and old['code'] == code_hash('baseline') and 'reused_legacy_products' in old and old.get('numerical_threads')==1:
            for m in ('CLAP', 'PULP'):
                if digest(out / m / 'features.csv') != old['features'][m]: raise ValueError('特征缓存改变')
                load_features(out, m)
            return 'cached'
    rows = list(records(v))
    if any(not (r['coverage_agrees'] and r['source'] and r['test_source']) for r in rows):
        raise ValueError('观测数据不完整')
    atomic_json(out / 'inventory.json', rows)
    from .multifault_cache import seed
    reused = seed(v,before)
    slices(v, out, v.parent / '.fp_research/slice-cache')
    for m in ('CLAP', 'PULP'):
        features(v, out, m)
        names, obs, table = load_features(out, m)
        expected = {r['configuration'] for r in rows if m == 'PULP' or r['outcome'] == 'passed'}
        if set(names) != expected or len(names) != len(expected): raise ValueError('基线特征配置不符')
        actual = dict(zip(names, obs))
        if any((actual[r['configuration']] == 'F') != (r['outcome'] == 'failed') for r in rows if r['configuration'] in actual):
            raise ValueError('基线可见标签不符')
    if raw_digest(v) != before: raise ValueError('基线准备改变原始文件')
    atomic_json(stamp, dict(raw_digest=before, code=code_hash('baseline'), reused_legacy_products=reused,numerical_threads=1,
                           features={m: digest(out / m / 'features.csv') for m in ('CLAP', 'PULP')}))
    return 'verified'


def partitions(jobs):
    identities = {folder(j): (j['system'], j['version']) for j in jobs}
    splits = []
    for system, stratum in sorted({(j['system'], j['stratum']) for j in jobs}):
        selected = [j for j in jobs if (j['system'], j['stratum']) == (system, stratum)]
        selected.sort(key=lambda j: hashlib.sha256(('42/'+system+'/'+j['version']).encode()).digest())
        ordered = [folder(j) for j in selected]
        for fold in range(5):
            testing = ordered[fold::5]
            splits.append((f'{system}-{stratum}fault-fold{fold}', [p for p in ordered if p not in testing], testing))
    return splits, identities


def labels(jobs):
    # 只在全部案例预测保存之后读取真值；不向联合推断传入标签。
    from .truth import ground_truth
    for j in jobs:
        if not (folder(j) / 'joint.json').exists(): raise ValueError('先完成全部联合推断再生成评价标签')
    values = []
    for system, stratum in sorted({(j['system'], j['stratum']) for j in jobs}):
        selected = [j for j in jobs if (j['system'], j['stratum']) == (system, stratum)]
        root = Path(selected[0]['path']).parent
        rows = list(ground_truth(root.parent, {system: (root.name,)}))
        any_base={}
        for j in selected:
            v=Path(j['path'])
            features={line.split(':',1)[0].split('.',1)[0] for line in (v/(v.name+'.mutant.log')).read_text(encoding='utf8').splitlines()}
            any_base[j['version']]=int(bool(features & {'BankAccount','Base','ExamDB'}))
        expected = {(j['version'], r['configuration']) for j in selected for r in records(Path(j['path'])) if r['outcome']=='passed'}
        if len(rows) != len(expected) or {(r['version'], r['configuration']) for r in rows} != expected:
            raise ValueError('评价真值清单不符')
        values.extend(dict(**r, stratum=stratum, any_base=any_base[r['version']]) for r in rows)
    write_csv(OUTPUT / 'truth.csv', values)


def classify(jobs, method, original_default=False):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    import pandas as pd
    from .compare import run, load_features
    if original_default and method!='PULP':raise ValueError('默认抽样对照仅用于PULP')
    output_name='PULP-default' if original_default else method
    mapping = {}
    if method == 'CLAP':
        truth = pd.read_csv(OUTPUT / 'truth.csv')
        mapping = dict(zip(truth[KEYS].itertuples(index=False, name=None), truth.label))
    splits, identities = partitions(jobs)
    # 各折即时归档，避免完成几十折之后因一折中断而丢失全部输出。
    parts = []; training_receipts = []
    for split in splits:
        target = OUTPUT / 'fold_predictions' / f'{output_name}-{split[0]}.csv'
        observed = []; train_labels = []; feature_inputs = []
        for path in split[1]:
            ids, obs, table = load_features(path, method)
            observed.extend(obs)
            if method == 'CLAP': train_labels.extend(mapping[(*identities[path], cid)] for cid in ids)
            feature_inputs.append(dict(system=identities[path][0], version=identities[path][1],
                                       sha256=digest(path/method/'features.csv'), configurations=ids))
        run(None, ROOT.parent / method, method, mapping, target, partitions=[split], identities=identities,
            pu_sampling_ratio=.05 if original_default else None)
        positive=observed.count('F'); unlabeled=len(observed)-positive
        ratio=.05 if original_default else (positive+1e-6)/unlabeled if method=='PULP' else None
        training_receipts.append(dict(split=split[0], method=output_name, rows=len(observed),
            columns=6 if method=='CLAP' else 50, known_failure_positives=positive if method=='PULP' else None,
            training_true_positives=sum(train_labels) if method=='CLAP' else None,
            sampling_ratio=ratio,
            sampled_unlabeled_per_estimator=int(unlabeled*ratio) if method=='PULP' else None,
            estimators=30 if method=='PULP' else 1, seed=42 if method=='PULP' else None, input_files=feature_inputs,
            testing=[dict(system=identities[p][0],version=identities[p][1]) for p in split[2]], output_sha256=digest(target)))
        atomic_json(OUTPUT / f'{output_name}-training.json', training_receipts)
        parts.append(pd.read_csv(target))
    combined=pd.concat(parts, ignore_index=True)
    if original_default:combined['method']=output_name
    combined.to_csv(OUTPUT / f'{output_name}.csv', index=False)


def evaluate(jobs):
    import pandas as pd
    from .extension_evaluate import eval_metrics
    truth = pd.read_csv(OUTPUT / 'truth.csv')
    splits, identities = partitions(jobs)
    assigned = {identities[p]: name for name, _, test in splits for p in test}
    joint = pd.concat(pd.read_csv(folder(j) / 'joint.csv') for j in jobs)
    joint['split'] = [assigned[k] for k in zip(joint.system, joint.version)]
    parts = [joint.drop(columns='stratum'), *[pd.read_csv(OUTPUT / f'{m}.csv') for m in ('CLAP', 'PULP')]]
    frame = pd.concat(parts, ignore_index=True).merge(truth, on=KEYS, validate='many_to_one', how='left')
    expected = set(truth[KEYS].itertuples(index=False, name=None))
    for method, part in frame.groupby('method'):
        if len(part) != len(expected) or set(part[KEYS].itertuples(index=False, name=None)) != expected:
            raise ValueError('三方法候选分母不同或有重复: '+method)
    if frame.label.isna().any(): raise ValueError('缺少评价标签')
    assignments = []
    for name, training, testing in splits:
        if set(training) & set(testing): raise ValueError('训练和测试版本重叠')
        assignments.extend(dict(split=name, role=role, system=identities[p][0], version=identities[p][1])
                           for role, paths in [('train', training), ('test', testing)] for p in paths)
    write_csv(OUTPUT / 'folds.csv', assignments)
    tables = []
    for (system, stratum, method), part in frame.groupby(['system', 'stratum', 'method']):
        for subset, sub, label in [('all', part, 'label'), ('executed', part, 'executed_label'),
                                    ('all_base', part[part.base==1], 'label'), ('other', part[part.base==0], 'label'),
                                    ('any_base', part[part.any_base==1], 'label'), ('no_base', part[part.any_base==0], 'label')]:
            if len(sub): tables.append(dict(system=system, stratum=int(stratum), method=method, subset=subset,
                unstable=int((sub.stable==0).sum()) if method=='joint' else None,
                ambiguous=int(((sub.lower<.5)&(sub.upper>=.5)).sum()) if method=='joint' else None,
                **eval_metrics(sub, label)))
    frame.to_csv(OUTPUT / 'predictions.csv', index=False)
    pd.DataFrame(tables).to_csv(OUTPUT / 'metrics.csv', index=False)
    versions = []
    for j in jobs:
        d = read(folder(j) / 'joint.json')
        versions.append(dict(**j, stable=d['stable'], seconds=d['seconds'], classes=d['classes'], tests=d['tests'],
                             particles=d['attempts'][-1]['particles'], seed_difference=d['attempts'][-1]['difference']))
    write_csv(OUTPUT / 'versions.csv', versions)
    print(pd.DataFrame(tables).query("subset=='all'").to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['manifest', 'fit', 'prepare', 'labels', 'classify', 'evaluate'])
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--smoke', action='store_true', help='每系统每缺陷层仅验证第一个案例')
    p.add_argument('--system', choices=['elevator', 'email', 'examdb'])
    p.add_argument('--version')
    p.add_argument('--method', choices=['CLAP', 'PULP'])
    p.add_argument('--original-default',action='store_true',help='额外运行PULP源码默认5%抽样，单独归档')
    a = p.parse_args(); jobs = manifest()
    if a.stage in {'fit', 'prepare'}:
        if a.system: jobs = [j for j in jobs if j['system'] == a.system]
        if a.version: jobs = [j for j in jobs if j['version'] == a.version]
        if a.smoke:
            jobs = [next(j for j in jobs if (j['system'], j['stratum']) == key) for key in sorted({(j['system'], j['stratum']) for j in jobs})]
        executor = ProcessPoolExecutor if a.stage == 'fit' else ThreadPoolExecutor
        action = fit if a.stage == 'fit' else prepare
        failures = []
        with executor(max_workers=a.workers) as pool:
            tasks = {pool.submit(action, j): j for j in jobs}
            for future in as_completed(tasks):
                j = tasks[future]; error = folder(j) / f'{a.stage}-failure.json'
                try:
                    result = future.result()
                    error.unlink(missing_ok=True)
                    print(j['system'], j['version'], result, flush=True)
                except Exception:
                    message = traceback.format_exc()
                    atomic_json(error, dict(error=message))
                    failures.append(j)
                    print('FAILED', j['system'], j['version'], message, flush=True)
        if failures: raise RuntimeError(f'{len(failures)} 个案例待修复；保留全部失败记录')
    elif a.stage == 'labels': labels(jobs)
    elif a.stage == 'classify':
        if not a.method: p.error('分类需指定 --method')
        classify(jobs, a.method, a.original_default)
    elif a.stage == 'evaluate': evaluate(jobs)
    else: print(json.dumps(dict(versions=len(jobs), configurations=sum(j['configurations'] for j in jobs), candidates=sum(j['candidates'] for j in jobs))))


if __name__ == '__main__': main()
