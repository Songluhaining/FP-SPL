"""从逐次补测记录独立复算互斥状态、预算和两次确认数量。"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from fp_research.case_study import ROOT
from fp_research.inventory import atomic_json


def main():
    folder=ROOT/'experimental_results/gpl_zipme/main'
    def read(name):
        with (folder/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    raw=read('replay-outcomes.csv')
    identities=[(r['system'],r['version'],r['configuration'],r['donor_test']) for r in raw]
    if len(set(identities))!=len(identities):raise ValueError('重复执行记录')
    for r in raw:
        if r['status']=='confirmed':
            reference=json.loads(r['reference']);first=json.loads(r['first']);second=json.loads(r['second'])
            if reference[0]!='passed' or first[0]!='assertion' or first!=second:raise ValueError('确认条件不成立')
    statuses=read('replay-status-counts.csv')
    tables=read('replay-budget-counts.csv')
    if len(tables)!=6:raise ValueError('系统或预算缺失')
    for row in tables:
        system,budget=row['system'],int(row['budget'])
        selected=[r for r in raw if r['system']==system and int(r['position'])<=budget]
        counts=Counter(r['status'] for r in selected)
        archived={r['status']:int(r['tests']) for r in statuses if r['system']==system and int(r['budget'])==budget}
        if counts!=archived or sum(counts.values())!=int(row['planned_tests']):raise ValueError('互斥状态未覆盖全部执行')
        configurations={(r['version'],r['configuration']) for r in selected}
        confirmed={(r['version'],r['configuration']) for r in selected if r['status']=='confirmed'}
        expected=dict(with_tests=len(configurations),confirmed=len(confirmed),compile_errors=counts['compile_error'],
                      timeouts=sum(n for s,n in counts.items() if 'timeout' in s),
                      runtime_errors=sum(n for s,n in counts.items() if s.endswith(('runtime_error','runner_error'))),
                      reference_assertion_failures=counts['reference_assertion'])
        if any(int(row[k])!=v for k,v in expected.items()):raise ValueError('预算统计复算不一致')
    files=['replay-outcomes.csv','replay-budget-counts.csv','replay-status-counts.csv']
    atomic_json(folder/'replay-statistics-verified.json',dict(tests=len(raw),budget_rows=len(tables),
        files={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in files}))
    print(len(raw),'条补测记录与6组预算统计独立核验通过',flush=True)


if __name__=='__main__':main()
