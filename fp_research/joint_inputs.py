"""只从可见配置、源码映射和逐测试覆盖构建联合模型；不读取变异信息。"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .inventory import records,atomic_json
from .latent_fault import read_version
from .joint_fault import JointEvidence


def load_evidence(version):
    version=Path(version)
    aggregate=read_version(version)
    inventory=list(records(version))
    if any(not (r['coverage_agrees'] and r['source'] and r['test_source']) for r in inventory):
        raise ValueError('原始数据不完整')
    rawhash=hashlib.sha256()
    source_files=[]
    def read(path):
        value=path.read_bytes()
        relative=path.relative_to(version).as_posix()
        digest=hashlib.sha256(value).hexdigest()
        source_files.append(dict(path=relative,sha256=digest))
        rawhash.update(relative.encode());rawhash.update(value)
        return value
    read(version/'config.report.csv.done')
    observations=[]
    identity=[]
    for record,old in zip(inventory,aggregate,strict=True):
        if record['configuration']!=old.configuration:
            raise ValueError('配置顺序不一致')
        folder=version/'variants'/old.configuration/'coverage'
        for status in ('passed','failed'):
            path=folder/('spectrum_'+status+'_coverage.xml')
            if not path.exists():continue
            root=ET.fromstring(read(path))
            mapping={}
            for package in root.iterfind('.//package'):
                for file in package.findall('file'):
                    name=Path(file.attrib['path'].replace(chr(92),'/')).stem
                    for line in file.findall('line'):
                        if not line.get('featureClass') or not line.get('featureLineNum'):continue
                        key=(line.get('featureClass'),line.get('featureLineNum'))
                        local=(package.get('name'),name,line.attrib['num'])
                        if mapping.setdefault(local,key)!=key:raise ValueError('源位置映射冲突')
            count=Counter();files=sorted((folder/status).glob('*.coverage.xml'))
            expected_n=old.passed_tests if status=='passed' else old.failed_tests
            if len(files)!=expected_n:raise ValueError('逐测试文件数量不符')
            for file in files:
                reached=set()
                for package in ET.fromstring(read(file)).iterfind('.//package'):
                    for unit in package.findall('file'):
                        name=Path(unit.attrib['path'].replace(chr(92),'/')).stem
                        for line in unit.findall('line'):
                            if int(line.get('count',0))<=0:continue
                            key=mapping.get((package.get('name'),name,line.attrib['num']))
                            if key is not None and key in old.present:reached.add(key)
                count.update(reached)
                observations.append((frozenset(reached),int(status=='failed')))
                identity.append(dict(configuration=old.configuration,test=file.name,status=status))
            expected=old.passed_hits if status=='passed' else old.failed_hits
            if any(count[k]!=expected.get(k,0) for k in old.present):
                raise ValueError('逐测试覆盖与汇总不一致')
    # 候选只要求被某次失败执行到，不要求出现在所有失败配置。
    candidates=sorted(set().union(*(r for r,y in observations if y)))
    if not candidates:raise ValueError('失败测试没有可映射候选')
    index={k:i for i,k in enumerate(candidates)}
    matrix=np.zeros((len(observations),len(candidates)),dtype=np.uint8)
    labels=np.array([y for _,y in observations],dtype=np.uint8)
    for i,(reached,_) in enumerate(observations):
        for key in reached:
            if key in index:matrix[i,index[key]]=1
    # 这是严格观测等价，不是按距离强制聚类或指定故障数。
    equivalence=defaultdict(list)
    for j in range(len(candidates)):
        equivalence[np.packbits(matrix[:,j]).tobytes()].append(j)
    groups=sorted(equivalence.values(),key=lambda members:members[0])
    columns=matrix[:,[members[0] for members in groups]]
    patterns,inverse=np.unique(columns,axis=0,return_inverse=True)
    failed=np.bincount(inverse,weights=labels,minlength=len(patterns))
    passed=np.bincount(inverse,weights=1-labels,minlength=len(patterns))
    features=[k[0].split('.')[0] for k in candidates]
    counts=Counter(features)
    mass=np.array([1/(len(counts)*counts[f]) for f in features])
    present=np.array([[k in r.present for k in candidates] for r in aggregate],dtype=float)
    class_mass=np.array([mass[m].sum() for m in groups])
    membership=np.column_stack([present[:,m]@mass[m]/mass[m].sum() for m in groups])
    lower=np.column_stack([present[:,m].min(axis=1) for m in groups])
    upper=np.column_stack([present[:,m].max(axis=1) for m in groups])
    evidence=JointEvidence(patterns.astype(float),passed,failed,membership,lower,upper,class_mass)
    info=dict(configurations=[r.configuration for r in aggregate],
              passing=[r.failed_tests==0 for r in aggregate],
              tests=len(observations),patterns=len(patterns),candidate_statements=len(candidates),
              classes=[[':'.join(candidates[j]) for j in members] for members in groups],
              unidentifiable_classes=sum(len(m)>1 for m in groups),
              membership_ambiguous_configurations=int(np.any(lower!=upper,axis=1).sum()),
              fingerprint=rawhash.hexdigest(),input_files=source_files)
    return evidence,info


def save_evidence(version,e,info):
    folder=Path(version)/'.fp_research/joint_pilot'
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'evidence.npz',**{k:getattr(e,k) for k in e.__dataclass_fields__})
    atomic_json(folder/'evidence.json',info)
    return folder


def verify_unchanged(version,info):
    for item in info['input_files']:
        if hashlib.sha256((Path(version)/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('原始输入改变: '+item['path'])
