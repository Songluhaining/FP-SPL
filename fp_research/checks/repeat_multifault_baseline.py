"""各系统各缺陷层重训一个留出折，核对基线重复运行的逐配置结果。"""
import argparse
import pandas as pd
from fp_research.multifault_study import ROOT, OUTPUT, read, partitions, KEYS, digest
from fp_research.compare import run
from fp_research.inventory import atomic_json


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['CLAP','PULP','PULP-default'],required=True)
    method=p.parse_args().method
    jobs=read(OUTPUT/'manifest.json')['jobs'];splits,identities=partitions(jobs)
    mapping={}
    if method=='CLAP':
        labels=pd.read_csv(OUTPUT/'truth.csv')
        mapping=dict(zip(labels[KEYS].itertuples(index=False,name=None),labels.label))
    checks=[]
    for split in splits:
        if not split[0].endswith('fold0'):continue
        target=ROOT/'result/multifault_validation/repeated'/f'{method}-{split[0]}.csv'
        source_method='PULP' if method=='PULP-default' else method
        run(None,ROOT.parent/source_method,source_method,mapping,target,partitions=[split],identities=identities,
            pu_sampling_ratio=.05 if method=='PULP-default' else None)
        original=OUTPUT/'fold_predictions'/target.name
        pd.testing.assert_frame_equal(pd.read_csv(original),pd.read_csv(target),check_exact=True)
        checks.append(dict(split=split[0],method=method,rows=len(pd.read_csv(target)),
                           original_sha256=digest(original),repeated_sha256=digest(target),identical=True))
    if len(checks)!=6:raise ValueError('重复核验折不全')
    atomic_json(OUTPUT/f'{method}-repeat-check.json',checks)
    print(method,'六折独立重训逐值一致',flush=True)


if __name__=='__main__':main()
