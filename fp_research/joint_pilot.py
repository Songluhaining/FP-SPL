"""固定清单的多故障先导验证：先保存无真值推断，再单独读取评价真值。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .inventory import atomic_json
from .joint_fault import infer_joint,JointEvidence
from .joint_inputs import load_evidence,save_evidence,verify_unchanged
from .latent_fault import infer,read_version

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'experimental_results/multifault_pilot'
DATA=Path('D:/dataset/whn/BankAccountTP')
PARAMETERS=dict(particles=256,points=9,sparsity=1.5,sweeps=2)


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def manifest(count):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    rows=[];selected=[]
    for stratum in (1,2,3):
        folder=DATA/f'4wise-BankAccountTP-{stratum}BUG-Full'
        available=[]
        for version in sorted(folder.glob('_MultipleBugs_*')):
            with (version/'config.report.csv.done').open(encoding='utf-8-sig') as stream:
                configs=list(csv.reader(stream))[1:]
            passed=sum(r[-1]=='__PASSED__' for r in configs)
            rows.append(dict(stratum=stratum,version=version.name,configurations=len(configs),passed=passed))
            if 0<passed<len(configs):available.append(version)
        available.sort(key=lambda v:hashlib.sha256(('joint-pilot-20261113/'+v.name).encode()).digest())
        selected.extend(dict(stratum=stratum,version=v.name,path=str(v)) for v in available[:count])
    write_csv(OUTPUT/'inventory.csv',rows)
    spec=dict(selection='按名称加固定盐的摘要排序；仅需有可见失败和原本通过配置；未读取变异日志',
              selected=selected,parameters=PARAMETERS,seeds=[42,2026],
              stability_rule='两种种子在所有候选配置的最大分数差不超过0.05；否则增加粒子，仍不稳则标记',
              code={name:hashlib.sha256((ROOT/'fp_research'/name).read_bytes()).hexdigest()
                    for name in ('joint_fault.py','joint_inputs.py','joint_pilot.py')})
    target=OUTPUT/'manifest.json'
    if target.exists():
        old=json.loads(target.read_text(encoding='utf8'))
        if old['selected']!=spec['selected'] or old['parameters']!=spec['parameters']:
            raise ValueError('已有先导协议不符，不能静默覆盖')
    atomic_json(target,spec)
    return selected


def fit(job):
    started=time.monotonic();version=Path(job['path'])
    evidence,info=load_evidence(version)
    folder=save_evidence(version,evidence,info)
    runs=[]
    for seed in (42,2026):
        runs.append(infer_joint(evidence,seed=seed,**PARAMETERS))
    selected=np.array(info['passing'],dtype=bool)
    discrepancy=float(np.max(np.abs(runs[0]['score'][selected]-runs[1]['score'][selected])))
    if discrepancy>.05:
        runs=[infer_joint(evidence,seed=seed,**dict(PARAMETERS,particles=1024,sweeps=3)) for seed in (42,2026)]
        discrepancy=float(np.max(np.abs(runs[0]['score'][selected]-runs[1]['score'][selected])))
    old_error=None
    try:old={r['configuration']:r['score'] for r in infer(read_version(version),version.name) if r['method']=='statement'}
    except ValueError as exc:old={};old_error=str(exc)
    rows=[]
    mean={key:np.mean([r[key] for r in runs],axis=0) for key in ('score','lower','upper','inclusion')}
    independent=1-np.prod(1-evidence.membership*mean['inclusion'][None,:],axis=1)
    for i,name in enumerate(info['configurations']):
        if not selected[i]:continue
        rows.append(dict(configuration=name,score=float(mean['score'][i]),lower=float(mean['lower'][i]),
                         upper=float(mean['upper'][i]),independent_score=float(independent[i]),
                         old_score=old.get(name),prediction=int(mean['score'][i]>=.5),
                         robust_decision='positive' if mean['lower'][i]>=.5 else 'negative' if mean['upper'][i]<.5 else 'ambiguous'))
    diagnostics=dict(tests=info['tests'],patterns=info['patterns'],candidate_statements=info['candidate_statements'],
        classes=len(info['classes']),unidentifiable_classes=info['unidentifiable_classes'],
        membership_ambiguous_configurations=info['membership_ambiguous_configurations'],
        fingerprint=info['fingerprint'],score_seed_difference=discrepancy,stable=discrepancy<=.05,
        expected_mechanisms=float(np.mean([r['expected_mechanisms'] for r in runs])),old_error=old_error,
        posterior_class_inclusion=mean['inclusion'].tolist(),
        runs=[{k:r[k] for k in ('seed','particles','points','sparsity','trace','log_evidence','epsilon')} for r in runs])
    verify_unchanged(version,info)
    diagnostics['code']={name:hashlib.sha256((ROOT/'fp_research'/name).read_bytes()).hexdigest()
                         for name in ('joint_fault.py','joint_inputs.py','joint_pilot.py')}
    np.savez_compressed(folder/'posterior-states.npz',seed42=runs[0]['states'],seed2026=runs[1]['states'])
    diagnostics['seconds']=time.monotonic()-started
    atomic_json(folder/'inference.json',dict(predictions=rows,diagnostics=diagnostics))
    print(version.name,'classes',len(info['classes']),'patterns',info['patterns'],'seconds',round(diagnostics['seconds'],2),
          'seed_difference',round(discrepancy,4),flush=True)
    return dict(**job,**diagnostics)


def evaluate(selected):
    # 评价模块在全部推断保存以后才加载。模型入口仅有执行矩阵等可见证据。
    from .truth import ground_truth
    mapping={str(s):(f'BankAccountTP/4wise-BankAccountTP-{s}BUG-Full',) for s in (1,2,3)}
    chosen={(str(j['stratum']),j['version']) for j in selected}
    truth={}
    for r in ground_truth(DATA.parent,mapping):
        if (r['system'],r['version']) in chosen:truth[(int(r['system']),r['version'],r['configuration'])]=r
    predictions=[];summaries=[]
    for job in selected:
        data=json.loads((Path(job['path'])/'.fp_research/joint_pilot/inference.json').read_text(encoding='utf8'))
        diagnostics=data['diagnostics']
        for name,digest in diagnostics['code'].items():
            if hashlib.sha256((ROOT/'fp_research'/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('推断后实现已改变，须重新推断: '+name)
        summaries.append(dict(stratum=job['stratum'],version=job['version'],**{k:v for k,v in diagnostics.items() if k not in ('runs','posterior_class_inclusion','code')}))
        for row in data['predictions']:
            key=(job['stratum'],job['version'],row['configuration'])
            label=truth.pop(key)
            predictions.append(dict(stratum=job['stratum'],version=job['version'],**row,label=label['label'],
                                    executed_label=label['executed_label'],stable=int(diagnostics['stable'])))
    if truth:raise ValueError('评价清单有遗漏')
    write_csv(OUTPUT/'predictions.csv',predictions)
    write_csv(OUTPUT/'versions.csv',summaries)
    # 常规点预测包含全部候选，另外明确报告界跨阈值与采样未稳定配置。
    metrics=[]
    for s in (1,2,3):
        part=[r for r in predictions if r['stratum']==s]
        for method,column in [('single','old_score'),('joint','score'),('independent','independent_score')]:
            confusion=Counter((r['label'],int(r[column]>=.5) if r[column] is not None else 0) for r in part)
            tp,fp,tn,fn=[confusion[k] for k in ((1,1),(0,1),(0,0),(1,0))]
            metrics.append(dict(stratum=s,method=method,n=len(part),scored=sum(r[column] is not None for r in part),
                tp=tp,fp=fp,tn=tn,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                f1=2*tp/max(1,2*tp+fp+fn),accuracy=(tp+tn)/len(part),
                ambiguous=sum(r['robust_decision']=='ambiguous' for r in part) if method=='joint' else None,
                unstable=sum(not r['stable'] for r in part) if method=='joint' else None))
    write_csv(OUTPUT/'metrics.csv',metrics)
    atomic_json(OUTPUT/'completion.json',dict(versions=len(selected),predictions=len(predictions),
        completed=datetime.now().astimezone().isoformat(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [OUTPUT/'predictions.csv',OUTPUT/'metrics.csv',OUTPUT/'versions.csv',OUTPUT/'manifest.json']}))
    print(json.dumps(metrics,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['manifest','fit','evaluate'])
    p.add_argument('--per-stratum',type=int,default=6)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--limit',type=int)
    a=p.parse_args();selected=manifest(a.per_stratum)
    if a.stage=='fit':
        jobs=selected[:a.limit] if a.limit else selected
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for result in pool.map(fit,jobs):pass
    if a.stage=='evaluate':evaluate(selected)


if __name__=='__main__':main()
