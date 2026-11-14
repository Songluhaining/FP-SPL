"""独立核验完整多缺陷实验并生成统一中文表；不使用模型的风险或统计函数。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import gzip
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from fp_research.inventory import atomic_json
from fp_research.multifault_study import ROOT, OUTPUT, SCENARIO, read, digest, code_hash, write_csv
from fp_research.joint_inputs import verify_unchanged
from fp_research.case_study import raw_digest
from fp_research.result_registry import check, NAMES, METRICS


def verify_case(job, indexed):
    v = Path(job['path']); out = v / '.fp_research' / SCENARIO
    info = read(out / 'evidence.json'); receipt = read(out / 'joint.json')
    if receipt['code'] != code_hash('joint'): raise ValueError('推断源码改变')
    if read(out / 'baselines.json')['code'] != code_hash('baseline'): raise ValueError('特征源码改变')
    if raw_digest(v) != read(out / 'baselines.json')['raw_digest']: raise ValueError('原始文件改变')
    verify_unchanged(v, info)
    for name, sha in receipt['artifacts'].items():
        if digest(out / name) != sha: raise ValueError('归档摘要不符')
    e = np.load(out / 'evidence.npz'); states = np.load(out / 'posterior-states.npz')
    estimates = []
    # 对每个配置逐粒子取活动位置集合，独立复算联合风险（不调用 risks）。
    for field in ('membership', 'lower_membership', 'upper_membership'):
        values = []
        for key in states.files:
            samples = states[key]
            values.append([np.mean([1-math.prod(1-float(row[k]) for k in np.flatnonzero(particle))
                                    for particle in samples]) for row in e[field]])
        estimates.append(np.mean(values, axis=0))
    mutants = set()
    for text in (v / (v.name+'.mutant.log')).read_text(encoding='utf8').splitlines():
        name, number, _ = text.split(':', 2); mutants.add(name.rsplit('.', 1)[0]+':'+number)
    with (v / 'config.report.csv.done').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream); header = next(reader); configs = list(reader)
    expected = set()
    for cells in configs:
        cid = cells[0].strip()
        if cells[-1].strip() != '__PASSED__': continue
        expected.add((job['system'], v.name, cid))
        features = {h for h, x in zip(header[1:-1], cells[1:-1]) if x.strip()=='T'}
        present = set(); hit = set()
        for line in ET.parse(v/'variants'/cid/'coverage/spectrum_passed_coverage.xml').iterfind('.//package/file/line'):
            feature = line.get('featureClass', '')
            if feature.split('.')[0] not in features: continue
            key = feature+':'+str(line.get('featureLineNum')); present.add(key)
            if any(int(line.get(a, 0)) > 0 for a in ('count','truecount','falsecount')): hit.add(key)
        for method in ('joint','CLAP','PULP'):
            saved = indexed[(job['system'], v.name, cid, method)]
            if int(saved['label']) != bool(mutants & present) or int(saved['executed_label']) != bool(mutants & hit):
                raise ValueError('独立重建评价真值不符')
        saved = indexed[(job['system'], v.name, cid, 'joint')]
        i = info['configurations'].index(cid)
        for field, values in zip(('score','lower','upper'), estimates):
            if not np.isclose(saved[field], values[i], atol=1e-12, rtol=0): raise ValueError('粒子风险复算不符')
        if int(saved['prediction']) != int(saved['score'] >= .5): raise ValueError('固定阈值不符')
    archive = OUTPUT / 'cases' / job['system'] / v.name
    archive.mkdir(parents=True, exist_ok=True)
    for name in ('joint.json','evidence.npz','posterior-states.npz','baselines.json','inventory.json'):
        shutil.copyfile(out/name, archive/name)
    with gzip.GzipFile(filename=str(archive/'evidence.json.gz'), mode='wb', mtime=0) as stream:
        stream.write((out/'evidence.json').read_bytes())
    for method in ('CLAP','PULP'):
        if digest(out/method/'features.csv') != read(out/'baselines.json')['features'][method]: raise ValueError('特征改变')
        shutil.copyfile(out/method/'features.csv', archive/(method+'-features.csv'))
    print('verified',job['system'],v.name,flush=True)
    return expected


def main():
    jobs = read(OUTPUT/'manifest.json')['jobs']
    frame = pd.read_csv(OUTPUT/'predictions.csv')
    keys = ['system','version','configuration']
    indexed = {tuple(r[k] for k in keys+['method']): r for r in frame.to_dict('records')}
    if len(indexed) != len(frame): raise ValueError('预测重复')
    with ThreadPoolExecutor(max_workers=3) as pool:
        expected = set().union(*pool.map(lambda j: verify_case(j,indexed), jobs))
    for method, part in frame.groupby('method'):
        if set(part[keys].itertuples(index=False,name=None)) != expected or len(part)!=len(expected):
            raise ValueError('原始候选分母不符')
    if set(frame.method) != {'joint','CLAP','PULP'}: raise ValueError('缺少方法')
    folds = pd.read_csv(OUTPUT/'folds.csv')
    for name, part in folds.groupby('split'):
        tr = set(part.loc[part.role=='train',['system','version']].itertuples(index=False,name=None))
        te = set(part.loc[part.role=='test',['system','version']].itertuples(index=False,name=None))
        if tr & te: raise ValueError('版本交叉')
        predicted = frame[frame.split==name]
        if set(zip(predicted.system,predicted.version)) != te: raise ValueError('折归属不符')
    for m in ('CLAP','PULP'):
        for s,v,c,split in frame.loc[frame.method==m,keys+['split']].itertuples(index=False,name=None):
            if len(folds[(folds.system==s)&(folds.version==v)&(folds.role=='test')])!=1:
                raise ValueError('配置未恰好获得一次折外预测')
    metrics = pd.read_csv(OUTPUT/'metrics.csv')
    for row in metrics.to_dict('records'):
        part = frame[(frame.system==row['system'])&(frame.stratum==row['stratum'])&(frame.method==row['method'])]
        if row['subset']=='all_base': part=part[part.base==1]
        elif row['subset']=='other': part=part[part.base==0]
        check(part,row)
    now = datetime.now().astimezone().isoformat()
    chinese = []
    for row in metrics.to_dict('records'):
        selected = [j for j in jobs if j['system']==row['system'] and j['stratum']==row['stratum']]
        chinese.append(dict(系统=NAMES[row['system']],缺陷数量层=int(row['stratum']),
            方法={'joint':'联合贝叶斯诊断','CLAP':'有监督分类（CLAP）','PULP':'正例与未标注学习（PULP）'}[row['method']],
            实验场景='原本通过配置',版本划分='本案例独立推断' if row['method']=='joint' else '系统内同缺陷层按版本五折',
            评价子集={'all':'全部','executed':'缺陷行实际执行标签','all_base':'全部故障均在基础特征','other':'至少一个故障在其他特征'}[row['subset']],
            完成状态='已完成并独立复算核验',总缺陷版本数=len(selected),全部配置数=sum(j['configurations'] for j in selected),
            **{zh:row[k] for k,zh in METRICS.items()},
            原始通过配置数=sum(j['candidates'] for j in selected),
            结果来源=str(OUTPUT/'metrics.csv'),预测明细来源=str(OUTPUT/'predictions.csv'),
            来源文件摘要=digest(OUTPUT/'predictions.csv'),更新时间=now,
            说明='相对基准标签的识别结果；未进行实际补测确认；联合模型不跨案例训练；基线五折不等于基础变异未见。'))
    write_csv(OUTPUT/'三系统多缺陷实验比较.csv',chinese)
    old = pd.read_csv(ROOT/'experimental_results/六系统单缺陷实验比较.csv',encoding='utf-8-sig')
    old.insert(1,'缺陷数量层',1)
    pd.concat([old,pd.DataFrame(chinese)],ignore_index=True).to_csv(ROOT/'experimental_results/六系统单缺陷及多缺陷实验比较.csv',index=False,encoding='utf-8-sig')
    atomic_json(OUTPUT/'verified.json',dict(versions=len(jobs),candidates=len(expected),predictions=len(frame),
        independent_labels=True,particle_risks=True,raw_inputs_unchanged=True,folds_disjoint=True,
        metrics_sha256=digest(OUTPUT/'metrics.csv'),predictions_sha256=digest(OUTPUT/'predictions.csv'),completed=now))
    print('独立核验通过',len(jobs),len(expected),len(frame),flush=True)


if __name__=='__main__': main()
