"""独立核验完整多缺陷实验并生成统一中文表；不使用模型的风险或统计函数。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import gzip
import hashlib
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
    if receipt.get('numerical_threads')!=1 or read(out/'baselines.json').get('numerical_threads')!=1:
        raise ValueError('数值计算线程设置不符')
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
            if int(saved['any_base'])!=any(k.split('.',1)[0] in {'BankAccount','Base','ExamDB'} for k in mutants):
                raise ValueError('含基础特征缺陷的分层不符')
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
    # 独立按固定摘要重建分折，不调用实验入口的分折函数。
    for system,stratum in sorted({(j['system'],j['stratum']) for j in jobs}):
        ordered=sorted([j['version'] for j in jobs if j['system']==system and j['stratum']==stratum],
                       key=lambda name:hashlib.sha256(('42/'+system+'/'+name).encode()).digest())
        for number in range(5):
            name=f'{system}-{stratum}fault-fold{number}'
            actual=set(folds.loc[(folds.split==name)&(folds.role=='test'),'version'])
            if actual!=set(ordered[number::5]):raise ValueError('固定分折独立重建不符')
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
        receipts=read(OUTPUT/f'{m}-training.json')
        if len(receipts)!=30:raise ValueError('基线训练折记录不全')
        by_version={(j['system'],j['version']):j for j in jobs}
        for record in receipts:
            expected_train=set(folds.loc[(folds.split==record['split'])&(folds.role=='train'),['system','version']].itertuples(index=False,name=None))
            actual_train={(r['system'],r['version']) for r in record['input_files']}
            if actual_train!=expected_train:raise ValueError('实际训练输入与分折不符')
            n=positive=0
            for item in record['input_files']:
                job=by_version[(item['system'],item['version'])]
                path=Path(job['path'])/'.fp_research'/SCENARIO/m/'features.csv'
                if digest(path)!=item['sha256']:raise ValueError('训练后特征改变')
                table=pd.read_csv(path); names=table['VARIANT' if m=='CLAP' else 'model'].tolist()
                if names!=item['configurations']:raise ValueError('训练行序或配置不符')
                n+=len(table);positive+=int((table.LABEL=='F').sum())
            if n!=record['rows'] or record['columns']!=(6 if m=='CLAP' else 50):raise ValueError('训练规模不符')
            if m=='PULP':
                if positive!=record['known_failure_positives'] or not math.isclose(record['sampling_ratio'],(positive+1e-6)/(n-positive),abs_tol=1e-12):
                    raise ValueError('正例或抽样比例不符')
                if record['sampled_unlabeled_per_estimator']!=positive or record['estimators']!=30:raise ValueError('装袋抽样设置不符')
            output=OUTPUT/'fold_predictions'/f"{m}-{record['split']}.csv"
            if digest(output)!=record['output_sha256']:raise ValueError('基线预测改变')
    for row in pd.read_csv(OUTPUT/'baseline-source-hashes.csv').to_dict('records'):
        if digest(ROOT.parent/row['method']/row['path'])!=row['sha256']:raise ValueError('比较方法源码改变')
    metrics = pd.read_csv(OUTPUT/'metrics.csv')
    for row in metrics.to_dict('records'):
        part = frame[(frame.system==row['system'])&(frame.stratum==row['stratum'])&(frame.method==row['method'])]
        if row['subset']=='all_base': part=part[part.base==1]
        elif row['subset']=='other': part=part[part.base==0]
        elif row['subset']=='any_base': part=part[part.any_base==1]
        elif row['subset']=='no_base': part=part[part.any_base==0]
        check(part,row)
        if row['method']=='joint':
            if row['unstable']!=int((part.stable==0).sum()) or row['ambiguous']!=int(((part.lower<.5)&(part.upper>=.5)).sum()):
                raise ValueError('采样稳定性或识别区间统计不符')
    now = datetime.now().astimezone().isoformat()
    chinese = []
    for row in metrics.to_dict('records'):
        selected = [j for j in jobs if j['system']==row['system'] and j['stratum']==row['stratum']]
        chinese.append(dict(系统=NAMES[row['system']],缺陷数量层=int(row['stratum']),
            方法={'joint':'联合贝叶斯诊断','CLAP':'有监督分类（CLAP）','PULP':'正例与未标注学习（PULP，沿用等量抽样）'}[row['method']],
            实验场景='原本通过配置',版本划分='本案例独立推断' if row['method']=='joint' else '系统内同缺陷层按版本五折',
            评价子集={'all':'全部','executed':'缺陷行实际执行标签','all_base':'全部故障均在基础特征',
                      'other':'至少一个故障在其他特征','any_base':'至少一个故障在基础特征','no_base':'没有基础特征故障'}[row['subset']],
            完成状态='已完成并独立复算核验',总缺陷版本数=len(selected),全部配置数=sum(j['configurations'] for j in selected),
            已完成准备版本数=len(selected),主实验观察核验完成版本数=len(selected),主实验补测完成版本数=0,
            **{zh:row[k] for k,zh in METRICS.items()},
            采样未稳定配置数=row['unstable'],区间跨阈值配置数=row['ambiguous'],
            原始通过配置数=sum(j['candidates'] for j in selected),
            结果来源=str(OUTPUT/'metrics.csv'),预测明细来源=str(OUTPUT/'predictions.csv'),
            来源文件摘要=digest(OUTPUT/'predictions.csv'),更新时间=now,
            结果归档提交='见 multifault_complete/archive.json',
            说明='相对基准标签的识别结果；未进行实际补测确认；联合模型不跨案例训练；基线五折不等于基础变异未见。'))
    write_csv(OUTPUT/'三系统多缺陷实验比较.csv',chinese)
    old = pd.read_csv(ROOT/'experimental_results/六系统单缺陷实验比较.csv',encoding='utf-8-sig')
    old.insert(1,'缺陷数量层',1)
    pd.concat([old,pd.DataFrame(chinese)],ignore_index=True).to_csv(ROOT/'experimental_results/六系统单缺陷及多缺陷实验比较.csv',index=False,encoding='utf-8-sig')
    atomic_json(OUTPUT/'verified.json',dict(versions=len(jobs),candidates=len(expected),predictions=len(frame),
        independent_labels=True,particle_risks=True,raw_inputs_unchanged=True,folds_disjoint=True,
        metrics_sha256=digest(OUTPUT/'metrics.csv'),predictions_sha256=digest(OUTPUT/'predictions.csv'),completed=now))
    names={'joint':'联合贝叶斯诊断','CLAP':'有监督分类（CLAP）','PULP':'正例与未标注学习（PULP，沿用等量抽样）'}
    report=['# 三系统多缺陷实验报告','',
        f'本轮覆盖全部{len(jobs)}个双、三缺陷案例、{sum(j["configurations"] for j in jobs)}个配置；三种方法均以相同的{len(expected)}个原本通过配置为评价对象。以下为相对基准标签的识别结果，未做实际补测确认。','',
        '联合方法逐案例独立推断，不跨版本训练；两种比较方法在各系统、各缺陷层分别按版本五折评价。参数与阈值在读取评价标签前固定，详见[实验协议](实验协议.md)。五折结果不等同于基础变异完全未见的泛化结果。','',
        '|系统|缺陷数|方法|候选数|成功评分|精确率|召回率|调和平均值|准确率|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in metrics[metrics.subset=='all'].to_dict('records'):
        report.append('|'+ '|'.join([NAMES[row['system']],str(int(row['stratum'])),names[row['method']],str(int(row['n'])),str(int(row['scored'])),
            *[f'{row[k]*100:.2f}%' for k in ('precision','recall','f1','accuracy')]])+'|')
    unstable=int((frame.loc[frame.method=='joint','stable']==0).sum())
    insufficient=frame.groupby('method').score.apply(lambda s:int(s.isna().sum())).to_dict()
    insufficient_text='；'.join(f'{names[m]} {n}个' for m,n in insufficient.items())
    report.extend(['',f'证据不足（无法评分）的候选数：{insufficient_text}；联合方法采样分数差仍超过预设容差的候选数：{unstable}。缺失分数保留在分母内，按未报出计算，不赋予排序命中。',
        '', '完整混淆矩阵、实际执行标签、基础特征子集、平均精确率和逐案例前10%检查预算命中数保存在[中文比较表](三系统多缺陷实验比较.csv)。[逐配置明细](predictions.csv)保留每个候选的分数、决策、标签及归属验证折。',
        '', '独立核验已复算后验粒子风险和评价标签，核对原始文件未改动、候选无遗漏无重复、三方法分母一致、基线训练与测试版本无交叉、实际抽样比例及所有统计。切片复用前后六个代表案例的12份基线特征表逐值一致。',
        '', '联合方法允许多个故障同时存在，但仍假设故障独立触发失败；本轮结果不证明已经建模故障之间的抑制或促进。无法区分的语句位置保留风险上下界，该范围不是95%置信区间。',
        '', '历史六系统单缺陷结果与本轮多缺陷结果汇入仓库上一级的“六系统单缺陷及多缺陷实验比较.csv”。历史单缺陷中的“贝叶斯诊断”是此前四步版本；本轮明确标为“联合贝叶斯诊断”，没有覆盖历史分数。银行账户多缺陷仍为此前先导验证，图处理与压缩系统多缺陷按用户要求暂缓。',''])
    (OUTPUT/'实验报告.md').write_text('\n'.join(report),encoding='utf8')
    print('独立核验通过',len(jobs),len(expected),len(frame),flush=True)


if __name__=='__main__': main()
