"""等待本轮所有案例准备完毕，再依次生成标签、运行基线并独立验收。"""
import os
from pathlib import Path
import subprocess
import sys
import time

from .inventory import atomic_json
from .multifault_study import ROOT, OUTPUT, read, code_hash, folder, digest


def main():
    jobs=read(OUTPUT/'manifest.json')['jobs']
    expected={kind:code_hash(kind) for kind in ('joint','baseline')}
    work=ROOT/'result/multifault_validation';work.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    while True:
        counts={kind:0 for kind in expected};failures=[]
        for j in jobs:
            out=folder(j)
            for kind,marker in [('joint','joint.json'),('baseline','baselines.json')]:
                p=out/marker
                if p.exists():
                    receipt=read(p)
                    if receipt['code']==expected[kind] and receipt.get('numerical_threads')==1:counts[kind]+=1
            failures.extend(str(p) for p in out.glob('*-failure.json'))
        print('ready',counts,'/',len(jobs),flush=True)
        if failures:raise RuntimeError('案例存在失败记录：'+str(failures))
        if all(n==len(jobs) for n in counts.values()):break
        time.sleep(30)
    if any(code_hash(k)!=v for k,v in expected.items()):raise ValueError('等待期间实现改变，须重新核验')
    def run(module,*args):
        subprocess.run([sys.executable,'-m','fp_research.'+module,*args],cwd=ROOT,env=env,check=True)
    run('multifault_study','labels')
    processes=[]
    for method in ('CLAP','PULP'):
        log=(work/f'classify-{method}.log').open('w',encoding='utf8')
        process=subprocess.Popen([sys.executable,'-m','fp_research.multifault_study','classify','--method',method],
                                 cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append((method,process,log))
    failed=[]
    for method,process,log in processes:
        code=process.wait();log.close()
        if code:failed.append(method)
    if failed:raise RuntimeError('基线分类失败：'+str(failed))
    run('multifault_study','evaluate')
    run('checks.verify_multifault')
    atomic_json(OUTPUT/'completion.json',dict(versions=len(jobs),verified=digest(OUTPUT/'verified.json'),
        files={p.name:digest(p) for p in [OUTPUT/'predictions.csv',OUTPUT/'metrics.csv',OUTPUT/'实验报告.md',
                                       ROOT/'experimental_results/六系统单缺陷及多缺陷实验比较.csv']}))
    print('完整比较及独立核验完成',flush=True)


if __name__=='__main__':main()
