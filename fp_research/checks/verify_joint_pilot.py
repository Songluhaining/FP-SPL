"""不用统计入口复算先导结果，检查后验粒子、候选清单、标签和原始输入。"""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from fp_research.inventory import atomic_json
from fp_research.joint_pilot import OUTPUT,write_csv
from fp_research.joint_inputs import verify_unchanged


def read(path):
    with path.open(encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))


def main():
    manifest=json.loads((OUTPUT/'manifest.json').read_text(encoding='utf8'))
    predictions=read(OUTPUT/'predictions.csv')
    indexed={(r['version'],r['configuration']):r for r in predictions}
    if len(indexed)!=len(predictions):raise ValueError('候选重复')
    audit=[]
    for job in manifest['selected']:
        version=Path(job['path']);folder=version/'.fp_research/joint_pilot'
        info=json.loads((folder/'evidence.json').read_text(encoding='utf8'))
        verify_unchanged(version,info)
        evidence=np.load(folder/'evidence.npz')
        states=np.load(folder/'posterior-states.npz')
        inference=json.loads((folder/'inference.json').read_text(encoding='utf8'))
        if inference['diagnostics']['code']!=manifest['code']:raise ValueError('协议源码与实测不符')
        estimates=[]
        for key in ('membership','lower_membership','upper_membership'):
            estimate=[]
            for name in states.files:
                per_config=[]
                for membership in evidence[key]:
                    total=0.
                    for particle in states[name]:
                        survival=1.
                        for active,value in zip(particle>0,membership):
                            if active:survival*=1-float(value)
                        total+=1-survival
                    per_config.append(total/len(states[name]))
                estimate.append(per_config)
            estimates.append(np.mean(estimate,axis=0))
        mutants=set()
        for text in (version/(version.name+'.mutant.log')).read_text(encoding='utf8').splitlines():
            location,line,_=text.split(':',2)
            mutants.add(location.rsplit('.',1)[0]+':'+line)
        with (version/'config.report.csv.done').open(encoding='utf-8-sig') as stream:
            reader=csv.reader(stream);header=next(reader);configs=list(reader)
        expected={r[0] for r in configs if r[-1]=='__PASSED__'}
        actual={r['configuration'] for r in predictions if r['version']==version.name}
        if actual!=expected:raise ValueError('配置有遗漏或新增')
        failed_present=[];false_negatives=[]
        for i,row in enumerate(configs):
            selected={name for name,value in zip(header[1:-1],row[1:-1]) if value.strip()=='T'}
            present=set();executed=set()
            for status in ('passed','failed'):
                p=version/'variants'/row[0]/'coverage'/('spectrum_'+status+'_coverage.xml')
                if not p.exists():continue
                for line in ET.parse(p).iterfind('.//package/file/line'):
                    feature=line.get('featureClass','')
                    if feature.split('.')[0] not in selected:continue
                    key=feature+':'+str(line.get('featureLineNum'))
                    present.add(key)
                    if any(int(line.get(k,0))>0 for k in ('count','truecount','falsecount')):executed.add(key)
            if row[-1]=='__FAILED__':failed_present.append(present);continue
            saved=indexed[(version.name,row[0])]
            for key,value in zip(('score','lower','upper'),estimates):
                if not np.isclose(float(saved[key]),value[i],rtol=0,atol=1e-12):raise ValueError('粒子复算不符')
            label=int(bool(mutants&present))
            if label!=int(saved['label']) or int(bool(mutants&executed))!=int(saved['executed_label']):
                raise ValueError(f'独立标签不一致 {version.name}/{row[0]}: present={label}/{saved["label"]}, executed={int(bool(mutants&executed))}/{saved["executed_label"]}, faults={sorted(mutants&present)}')
            if label and not int(saved['prediction']):
                false_negatives.append(dict(configuration=row[0],present_faults=sorted(mutants&present),score=float(saved['score'])))
        intersection=set.intersection(*failed_present)
        supported={k for group in info['classes'] for k in group}
        audit.append(dict(stratum=job['stratum'],version=version.name,faults=len(mutants),
            faults_excluded_by_intersection=sorted(mutants-intersection),
            faults_without_failed_execution=sorted(mutants-supported),false_negatives=false_negatives,
            raw_input_files=len(info['input_files'])))
        archive=OUTPUT/'cases'/version.name;archive.mkdir(parents=True,exist_ok=True)
        for name in ('evidence.npz','posterior-states.npz','inference.json'):
            shutil.copyfile(folder/name,archive/name)
        with gzip.GzipFile(filename=str(archive/'evidence.json.gz'),mode='wb',mtime=0) as stream:
            stream.write((folder/'evidence.json').read_bytes())
    for metric in read(OUTPUT/'metrics.csv'):
        rows=[r for r in predictions if r['stratum']==metric['stratum']]
        field={'single':'old_score','joint':'score','independent':'independent_score'}[metric['method']]
        counts=Counter((int(r['label']),int(float(r[field])>=.5) if r[field] else 0) for r in rows)
        tp,fp,tn,fn=[counts[k] for k in ((1,1),(0,1),(0,0),(1,0))]
        expected=dict(n=len(rows),scored=sum(bool(r[field]) for r in rows),tp=tp,fp=fp,tn=tn,fn=fn,
            precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),accuracy=(tp+tn)/len(rows))
        for key,value in expected.items():
            if not np.isclose(float(metric[key]),value,atol=1e-12,rtol=0):raise ValueError('指标复算不一致')
    chinese=[]
    for r in read(OUTPUT/'metrics.csv'):
        chinese.append(dict(zip(
            ['缺陷数量层','方法','版本数','候选数','检出','误报','正确排除','漏报','精确率','召回率','调和平均值','准确率','区间跨阈值数','采样未稳定数','说明'],
            [r['stratum'],{'single':'原四步单故障方法','joint':'联合后验研究原型','independent':'独立边缘风险近似'}[r['method']],
             sum(j['stratum']==int(r['stratum']) for j in manifest['selected']),
             *[r[k] for k in ('n','tp','fp','tn','fn','precision','recall','f1','accuracy','ambiguous','unstable')],
             '固定18版本的开发性先导实验；相对基准标签；不是六系统完整对比；比例为0至1。'])))
    write_csv(OUTPUT/'先导实验比较.csv',chinese)
    atomic_json(OUTPUT/'fault-support-audit.json',audit)
    atomic_json(OUTPUT/'verified.json',dict(versions=len(audit),configurations=len(predictions),
        independent_labels='通过',particle_risk_recalculation='通过',raw_inputs='未改变',
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUTPUT/'predictions.csv',OUTPUT/'metrics.csv')}))
    print('独立核验通过',len(audit),'版本',len(predictions),'候选配置',flush=True)


if __name__=='__main__':main()
