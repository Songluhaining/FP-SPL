"""实验常量及预测来源；不训练模型，不负责修改原始报告。"""
from functools import lru_cache
from pathlib import Path
import hashlib

from ..config import ROOT, datasets
from ..data.catalog import select_jobs
from ..io import read, digest

HISTORY = ROOT / 'experimental_results/statement_localization'
UPSTREAM = '62b413b52bbfba1f86a244ddb9f1daed232e329e'
METRICS = ['Tarantula','Ochiai','Op2','Barinel','Dstar','Jaccard']
METHODS = ['original','joint','CLAP','PULP','PULP-default']
SYSTEMS = dict(bank='银行账户',elevator='电梯',email='邮件',examdb='考试数据库',gpl='图处理',zipme='压缩')
NAMES = dict(original='未修正',joint='联合贝叶斯修正',CLAP='CLAP修正',PULP='PULP等量抽样修正',**{'PULP-default':'PULP默认5%修正'})


def jobs(system='all',stratum='all',version=None):
    return select_jobs(datasets(),system,stratum,version)


def historical_work_folder(job):
    return Path(job['path'])/'.fp_research/statement_localization'


def inference_folder(job):
    case=Path(job['path'])
    if job.get('detection_run'):
        return case/'.fp_research/manual_joint_detection'/job['detection_run']
    if job['stratum']>1 and job['system'] in {'elevator','email','examdb'}:
        return case/'.fp_research/multifault_joint'
    return historical_work_folder(job)


@lru_cache(maxsize=1)
def archived_fold_names():
    rows=read(HISTORY/'manifest.json')['jobs']
    result={}
    for system,stratum in sorted({(j['system'],j['stratum']) for j in rows}):
        selected=[j for j in rows if (j['system'],j['stratum'])==(system,stratum)]
        selected.sort(key=lambda j:hashlib.sha256(('42/'+system+'/'+j['version']).encode()).digest())
        for i,job in enumerate(selected):
            result[(system,job['version'])]=f'{system}-{stratum}fault-fold{i%5}'
    return result


def baseline_path(job,method):
    directory=Path(job.get('predictions_dir') or HISTORY/'fold_predictions')
    if job.get('predictions_dir'):
        matches=[]
        for path in directory.glob(f'{method}-*.json'):
            if (job['system'],job['version']) in set(map(tuple,read(path)['protocol']['testing'])):
                matches.append(path.with_suffix('.csv'))
        if len(matches)!=1: raise ValueError('指定预测目录必须为每个案例提供唯一折外预测')
        return matches[0]
    fold=archived_fold_names()[(job['system'],job['version'])]
    return directory/f'{method}-{fold}.csv'


def correction_sets(job):
    import csv
    import pandas as pd
    version=Path(job['path'])
    with (version/'config.report.csv.done').open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.reader(stream); header=next(reader); rows=list(reader)
    passed={r[0].strip() for r in rows if r[-1].strip()=='__PASSED__'}
    choices={'original':set()}; references={}; unavailable={}
    for method in METHODS[1:]:
        path=inference_folder(job)/'joint.csv' if method=='joint' else baseline_path(job,method)
        marker=path.with_name('joint.json') if method=='joint' else path.with_suffix('.json')
        proof=read(marker)
        if method=='joint':
            for name,sha in proof['artifacts'].items():
                if digest(path.parent/name)!=sha: raise ValueError('联合预测产物摘要改变')
            from ..detection.evidence import verify_unchanged
            verify_unchanged(version,read(path.parent/'evidence.json'))
            if job.get('detection_run'):
                from ..detection.runner import code_hash
                if proof['code']!=code_hash(): raise ValueError('指定的人工预测源码版本与当前不同')
        elif proof['sha256']!=digest(path):
            raise ValueError('折外预测摘要改变')
        data=pd.read_csv(path)
        if method!='joint': data=data[(data.system==job['system'])&(data.version==job['version'])]
        if set(data.configuration)!=passed or data.configuration.duplicated().any():
            raise ValueError(f'{method}的候选配置缺失或重复')
        choices[method]=set(data.loc[data.prediction==1,'configuration'])
        references[method]=dict(path=str(path),sha256=digest(path))
        unavailable[method]=int(data.score.isna().sum())
    return header,rows,choices,references,unavailable
