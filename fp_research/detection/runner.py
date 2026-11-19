"""每个案例独立推断；两种种子核对并逐级增加粒子。"""
from pathlib import Path
import time
import numpy as np
from ..config import ROOT
from ..io import read,digest,write_csv,atomic_json
from ..data.inventory import records
from .model import infer_joint
from .evidence import load_evidence,verify_unchanged
PARAMETERS=dict(particles=256,points=9,sparsity=1.5,sweeps=2)
SEEDS=(42,2026)
SCENARIO='manual_joint_detection'
def folder(job): return Path(job['path'])/'.fp_research'/SCENARIO
def code_hash(stage='joint'):
    if stage!='joint': raise ValueError('识别模块不负责基线特征')
    names=['detection/model.py','detection/evidence.py','data/coverage.py','data/inventory.py','detection/runner.py']
    return {n:digest(ROOT/'fp_research'/n) for n in names}
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
