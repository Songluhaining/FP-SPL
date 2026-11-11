"""核验提前运行的真实案例，并保存输入未变、缓存隔离和特征清单凭据。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from fp_research.case_study import ROOT,SYSTEMS,raw_digest,results
from fp_research.compare import load_features
from fp_research.inventory import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('D:/dataset/whn'))
    parser.add_argument('--version',action='append',required=True)
    a=parser.parse_args()
    system=a.data.joinpath(*SYSTEMS['zipme'])
    index=json.loads((system/'.fp_research/slice-signatures.json').read_text(encoding='utf8'))
    verified=[]
    for name in a.version:
        v=system/name
        out=results(v,'masked')
        complete=json.loads((out/'complete.json').read_text(encoding='utf8'))
        if raw_digest(v)!=complete['raw_digest']:raise ValueError('原始输入改变')
        rows=json.loads((out/'inventory.json').read_text(encoding='utf8'))
        original={r['configuration']:r for r in json.loads((results(v,'main')/'inventory.json').read_text(encoding='utf8'))}
        if len(rows)!=len(original) or {r['configuration'] for r in rows}!=set(original):raise ValueError('配置有遗漏或重复')
        audit=json.loads((out/'coverage-audit.json').read_text(encoding='utf8'))
        if audit['changed_coverage_counts'] or audit['tests']!=sum(len(ts) for r in rows for ts in r['tests'].values()):
            raise ValueError('逐用例覆盖核验不符')
        guard=json.loads((out/'concurrency-check.json').read_text(encoding='utf8'))
        pending=set(guard['unfinished_main_versions'])
        protected={e['raw'] for e in index['entries'] if Path(e['variant']).parent.parent.name in pending}
        if protected & {r['key'] for r in guard['checked']}:raise ValueError('样例与未完成主实验共享缓存键')
        if {r['configuration'] for r in guard['checked']}!={r['configuration'] for r in rows}:raise ValueError('缓存清单不完整')
        converted=[]
        for r in rows:
            previous=original[r['configuration']]
            if r['outcome']=='passed' and previous['outcome']=='failed':
                if r['tests']['failed'] or r['tests']['passed']!=previous['tests']['passed']:
                    raise ValueError('隐藏失败测试仍在输入中，或留存通过测试发生变化')
                converted.append(r['configuration'])
        feature_proofs=[]
        for method in ('CLAP','PULP'):
            names,observed,table=load_features(out,method)
            expected={r['configuration'] for r in rows if method=='PULP' or r['outcome']=='passed'}
            if len(names)!=len(expected) or set(names)!=expected:raise ValueError('特征行有遗漏或重复')
            labels={r['configuration']:('F' if r['outcome']=='failed' else 'TP') for r in rows}
            if any(labels[n]!=o for n,o in zip(names,observed)):raise ValueError('可见成败标签不符')
            feature_proofs.append(dict(method=method,rows=len(names),columns=len(table.columns),
                                      sha256=hashlib.sha256((out/method/'features.csv').read_bytes()).hexdigest()))
        verified.append(dict(version=name,configurations=len(rows),converted=converted,raw_digest=complete['raw_digest'],
                             no_pending_main_key_overlap=True,features=feature_proofs))
    atomic_json(ROOT/'experimental_results/gpl_zipme/overlap-verification.json',dict(
        generated=datetime.now().astimezone().isoformat(),
        runner_sha256=hashlib.sha256((ROOT/'fp_research/case_overlap.py').read_bytes()).hexdigest(),
        verified=verified,scope='真实案例调度与缓存隔离核验；不代表双系统分类评价已完成'))
    print(len(verified),'个真实补充案例核验通过',flush=True)


if __name__=='__main__':main()
