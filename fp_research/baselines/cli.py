"""比较方法入口：独立生成特征，按固定版本五折调用公开分类实现。"""
import argparse
import hashlib
import json
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

from ..config import ROOT,tool
from ..data.inventory import records
from ..experiments.catalog import jobs as select_jobs
from ..evaluation.partitions import partitions
from ..evaluation.truth import ground_truth
from ..io import read,digest,atomic_json,validate_run_name


def folder(job,run_name): return Path(job['path'])/'.fp_research/baselines'/run_name


def input_hashes(version):
    version=Path(version)
    files=[version/'config.report.csv.done']
    for variant in sorted((version/'variants').iterdir()):
        files.extend(sorted((variant/'src').rglob('*.java')))
        files.extend(sorted((variant/'test').rglob('*.java')))
        files.extend(sorted((variant/'coverage').rglob('*.xml')))
    return {str(p.relative_to(version)):digest(p) for p in files}


def prepare(job,run_name,method):
    from .slicing import slices
    from .classifier import load_features
    version=Path(job['path']); output=folder(job,run_name); output.mkdir(parents=True,exist_ok=True)
    before=input_hashes(version)
    rows=list(records(version))
    if any(not(r['source'] and r['test_source'] and r['coverage_agrees']) for r in rows):
        raise ValueError('原始输入不完整')
    slices(version,output,version.parent/'.fp_research/slice-cache')
    for name in (['CLAP','PULP'] if method=='all' else [method]):
        target=output/name; target.mkdir(exist_ok=True)
        env=os.environ.copy();env.update(PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with (output/f'{name}-run.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-m','fp_research.baselines.cli','worker','--case',str(version),
                '--output',str(output),'--method',name,'--run-name',run_name],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        load_features(output,name)
        upstream={str(p.relative_to(tool(name.lower()))):digest(p) for p in tool(name.lower()).rglob('*.py')}
        sources={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'fp_research/baselines').glob('*.py')}
        atomic_json(target/'features.json',dict(features_sha256=digest(target/'features.csv'),
            source_sha256=sources,upstream_python_tree_sha256=hashlib.sha256(json.dumps(upstream,sort_keys=True).encode()).hexdigest(),
            slicing_jar_sha256=digest(tool('pulp')/'plugins/all-statements-slicing.jar')))
    if input_hashes(version)!=before: raise ValueError('基线准备改变了原始输入')
    atomic_json(output/'inputs.json',before)
    print('特征完成',job['system'],job['version'],output,flush=True)


def classify(selected,run_name,method):
    from threadpoolctl import threadpool_limits
    from .classifier import run,load_features
    import pandas as pd
    threadpool_limits(limits=1)
    name=method; implementation='PULP' if name=='PULP-default' else name
    folders={(j['system'],j['version']):folder(j,run_name) for j in selected}
    parts,identities=partitions(selected,folders)
    labels={}
    if implementation=='CLAP':
        for system,stratum in sorted({(j['system'],j['stratum']) for j in selected}):
            first=next(j for j in selected if (j['system'],j['stratum'])==(system,stratum))
            layer=Path(first['path']).parent
            for row in ground_truth(layer.parent,{system:(layer.name,)}):
                labels[(system,row['version'],row['configuration'])]=row['label']
    output=ROOT/'experimental_results/baselines'/run_name
    directory=output/'fold_predictions';directory.mkdir(parents=True,exist_ok=True)
    sources=(['PassingVariants_Classification.py'] if implementation=='CLAP' else
             ['fp_detection/PUBaggingLearning.py','fp_detection/BaseClassifier.py'])
    tables=[];receipts=[]
    for split,training,testing in parts:
        inputs={};dimensions={};positive=total=0
        for path in training+testing:
            ids,observed,table=load_features(path,implementation)
            inputs[str(path/implementation/'features.csv')]=digest(path/implementation/'features.csv')
            feature_proof=read(path/implementation/'features.json')
            if feature_proof['features_sha256']!=inputs[str(path/implementation/'features.csv')]:
                raise ValueError('已准备的特征文件摘要改变')
            for source,sha in feature_proof['source_sha256'].items():
                if digest(ROOT/source)!=sha: raise ValueError('特征生成源码改变，请先重新准备特征')
            dimensions[str(path)]=dict(rows=len(ids),columns=list(table.columns))
            # 已保存的输入清单必须仍与原始输入一致。
            job=next(j for j in selected if identities[path]==(j['system'],j['version']))
            if read(path/'inputs.json')!=input_hashes(Path(job['path'])): raise ValueError('特征对应的原始输入已改变')
            if path in training: positive+=observed.count('F');total+=len(ids)
        target=directory/f'{name}-{split}.csv'
        run(tool(implementation.lower()),implementation,labels,target,[(split,training,testing)],identities,
            pu_sampling_ratio=.05 if name=='PULP-default' else None)
        ratio=(.05 if name=='PULP-default' else (positive+1e-6)/(total-positive)) if implementation=='PULP' else None
        spec=dict(split=split,method=name,training=[identities[p] for p in training],testing=[identities[p] for p in testing],
            input_files=inputs,tables=dimensions,sampling_ratio=ratio,
            proxy_negatives_per_estimator=int((total-positive)*ratio) if ratio is not None else None,
            estimators=30 if implementation=='PULP' else 1,classifier_sources={n:digest(tool(implementation.lower())/n) for n in sources},
            wrapper_sha256=digest(ROOT/'fp_research/baselines/classifier.py'))
        receipt=dict(protocol=spec,sha256=digest(target));atomic_json(target.with_suffix('.json'),receipt)
        receipts.append(receipt);tables.append(pd.read_csv(target))
    combined=pd.concat(tables,ignore_index=True)
    combined['method']=name
    if combined.duplicated(['system','version','configuration']).any(): raise ValueError('折外预测重复')
    combined.to_csv(output/f'{name}.csv',index=False)
    atomic_json(output/f'{name}-training.json',receipts)
    print('分类完成',name,output,flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','classify','worker'])
    parser.add_argument('--system',default='all',choices=['all','bank','email','elevator','examdb','gpl','zipme'])
    parser.add_argument('--stratum',default='1',choices=['1','2','3','all'])
    parser.add_argument('--version')
    parser.add_argument('--method',default='all',choices=['all','CLAP','PULP','PULP-default'])
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--case',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    validate_run_name(args.run_name)
    if args.stage=='worker':
        if args.method not in {'CLAP','PULP'} or not args.case or not args.output: parser.error('特征工作进程参数不完整')
        from .features import worker
        worker((tool(args.method.lower()),args.method,args.case.resolve(),args.output.resolve()/'slices',args.output.resolve()/args.method),optimize=True)
        return 0
    chosen=select_jobs(args.system,args.stratum,args.version)
    if args.stage=='prepare':
        if args.method=='PULP-default': args.method='PULP'
        for job in chosen: prepare(job,args.run_name,args.method)
    else:
        for method in (['CLAP','PULP','PULP-default'] if args.method=='all' else [args.method]): classify(chosen,args.run_name,method)
    return 0


if __name__=='__main__': raise SystemExit(main())
