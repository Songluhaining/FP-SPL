"""并行读取独立版本的原始输入；摘要算法与串行核验完全相同。"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json

from .case_study import ROOT,raw_digest,results
from .inventory import atomic_json


def verify_inputs(jobs,scenario,output,workers=8):
    if workers<1:raise ValueError('核验并行数必须为正')
    def one(job):
        system,version=job
        saved=json.loads((results(version,scenario)/'complete.json').read_text(encoding='utf8'))
        actual=raw_digest(version)
        if actual!=saved['raw_digest']:raise ValueError(f'完成后原始输入改变: {version}')
        return dict(system=system,version=version.name,raw_digest=actual)
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows)%20==0 or len(rows)==len(jobs):print('原始输入摘要核验',len(rows),'/',len(jobs),flush=True)
    identities={(r['system'],r['version']) for r in rows}
    if len(identities)!=len(jobs):raise ValueError('核验清单有重复版本')
    rows.sort(key=lambda r:(r['system'],r['version']))
    atomic_json(output,dict(workers=workers,scenario=scenario,
                           digest_source_sha256=hashlib.sha256((ROOT/'fp_research/case_study.py').read_bytes()).hexdigest(),
                           versions=rows))
    return {(r['system'],r['version']):r['raw_digest'] for r in rows}
