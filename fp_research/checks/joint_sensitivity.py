"""预先选定各层第一个案例，检查求积精度与稀疏先验；不据分数选择最优设置。"""
import json
from pathlib import Path
import numpy as np

from fp_research.inventory import atomic_json
from fp_research.joint_fault import JointEvidence,infer_joint
from fp_research.joint_pilot import OUTPUT,write_csv


def main():
    spec=json.loads((OUTPUT/'manifest.json').read_text(encoding='utf8'))
    jobs=[next(j for j in spec['selected'] if j['stratum']==s) for s in (1,2,3)]
    rows=[]
    for job in jobs:
        folder=Path(job['path'])/'.fp_research/joint_pilot'
        arrays=np.load(folder/'evidence.npz')
        e=JointEvidence(**{k:arrays[k] for k in arrays.files})
        info=json.loads((folder/'evidence.json').read_text(encoding='utf8'))
        previous=json.loads((folder/'inference.json').read_text(encoding='utf8'))['predictions']
        baseline=np.array([r['score'] for r in previous]);selected=np.array(info['passing'],dtype=bool)
        for points,sparsity in [(17,1.5),(33,1.5),(9,.5),(9,3.)]:
            r=infer_joint(e,particles=1024,points=points,sparsity=sparsity,seed=7701,sweeps=3)
            score=r['score'][selected]
            row=dict(stratum=job['stratum'],version=job['version'],points=points,sparsity=sparsity,
                particles=1024,max_score_difference=float(np.max(np.abs(score-baseline))),
                changed_predictions=int(np.sum((score>=.5)!=(baseline>=.5))),expected_mechanisms=r['expected_mechanisms'])
            rows.append(row);print(row,flush=True)
    write_csv(OUTPUT/'sensitivity.csv',rows)


if __name__=='__main__':main()
