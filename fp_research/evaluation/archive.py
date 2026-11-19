"""检查历史结果摘要；旧源码从对应Git提交核对，不恢复到当前工作树。"""
import argparse
import hashlib
from pathlib import Path
import subprocess

from ..config import ROOT
from ..io import read,digest


def verify_archive(manifest):
    manifest=Path(manifest)
    data=read(manifest)
    version=manifest.parent/'result_version.json'
    revision=read(version)['result_commit'] if version.is_file() else 'archive/pre-refactor-20261119'
    checked=historical_sources=0
    failures=[]
    for relative,expected in data['files'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT.resolve()): raise ValueError('归档路径超出仓库')
        if path.is_file() and digest(path)==expected:
            checked+=1;continue
        if not relative.replace('\\','/').startswith('experimental_results/'):
            blob=subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'show',revision+':'+relative.replace('\\','/')],
                                cwd=ROOT,capture_output=True)
            if blob.returncode==0 and hashlib.sha256(blob.stdout).hexdigest()==expected:
                checked+=1;historical_sources+=1;continue
        failures.append(relative)
    if failures: raise ValueError(f'{len(failures)}个归档文件缺失或摘要不符：{failures[:10]}')
    return dict(files=checked,historical_sources=historical_sources,source_revision=revision,passed=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'experimental_results/statement_localization_report_relabel/archive_manifest.json')
    args=parser.parse_args(argv)
    print(verify_archive(args.manifest))
    return 0


if __name__=='__main__': raise SystemExit(main())
