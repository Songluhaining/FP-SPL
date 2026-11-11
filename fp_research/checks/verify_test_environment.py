"""真实案例核对：兼容运行仍保留原有失败，移除后全部留存测试通过。"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from fp_research.case_study import JAVA,ROOT,raw_digest
from fp_research.inventory import atomic_json,records
from fp_research.runtime import LIB,Runtime
from fp_research.supplement import version_stage
from fp_research.test_environment import recompile_scaffolding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',type=Path,required=True)
    a=p.parse_args();v=a.version.resolve()
    out=v/'.fp_research/environment-check';out.mkdir(parents=True,exist_ok=True)
    before=raw_digest(v)
    selection=v/'.fp_research/masked/selection.csv'
    chosen={r['VARIANT'] for r in csv.DictReader(selection.open(encoding='utf-8-sig')) if r['FP transformed from F']=='1'}
    proofs=[]
    for row in records(v):
        if row['configuration'] not in chosen:continue
        with tempfile.TemporaryDirectory(dir=out) as temp:
            work=Path(temp).resolve();rt=Runtime(JAVA,work)
            variant=v/'variants'/row['configuration']
            main,tests,_=rt.compile(variant,work)
            mapping=recompile_scaffolding(rt,variant,work,main,tests)
            args=[str(rt.java),'-cp',os.pathsep.join([str(main),str(tests),LIB]),
                  'org.junit.platform.console.ConsoleLauncher','--details=summary','--disable-ansi-colors',
                  '--reports-dir',str(work/'reports')]
            for cls in sorted({n.rsplit('.',1)[0] for names in row['tests'].values() for n in names}):
                args+=['--select-class',cls]
            run=subprocess.run(args,cwd=work,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=120)
            (out/(row['configuration']+'.log')).write_text(run.stdout+run.stderr,encoding='utf8')
            seen=set();failed=set()
            for report in (work/'reports').glob('*.xml'):
                for test in ET.parse(report).iter('testcase'):
                    if test.find('skipped') is not None:continue
                    name=test.get('classname')+'.'+test.get('name').removesuffix('()')
                    seen.add(name)
                    if test.find('failure') is not None or test.find('error') is not None:failed.add(name)
            expected=set(row['tests']['failed'])
            if failed!=expected or seen!=set().union(*map(set,row['tests'].values())):
                raise ValueError(f"兼容运行与原报告不符 {row['configuration']} {failed ^ expected}")
            proofs.append(dict(configuration=row['configuration'],tests=len(seen),failed=sorted(failed),mapping=mapping))
            print(row['configuration'],len(seen),'原有失败精确保留',flush=True)
    with tempfile.TemporaryDirectory(dir=out) as temp:
        version_stage((v,Path(temp).resolve()/v.name,selection,out/'retained-tests.json',JAVA))
    checks=json.loads((out/'retained-tests.json').read_text(encoding='utf8'))
    if raw_digest(v)!=before or any(x['status']!='passed' for x in checks.values()):
        raise ValueError('输入摘要或留存测试核验失败')
    receipt=dict(version=v.name,raw_digest=before,original_failure_preserved=proofs,
                 retained_tests=checks,scope='仅映射运行副本中三个环境目录；原测试和产品不变')
    atomic_json(ROOT/'experimental_results/gpl_zipme/environment-verification.json',receipt)
    print('环境兼容核验通过',len(checks),flush=True)


if __name__=='__main__':main()
