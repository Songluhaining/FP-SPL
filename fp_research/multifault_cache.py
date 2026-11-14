"""复用输入完全匹配的既有切片；不复用标签、特征或模型输出。"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

from .inventory import atomic_json
from .slice_reuse import signatures
from .extension import slice_inputs
from .case_study import ROOT


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def index():
    config=json.loads((ROOT/'fp_research/dataset_locations.json').read_text(encoding='utf8'))
    jar=sha(ROOT.parent/'PULP/plugins/all-statements-slicing.jar')
    with (ROOT/'experimental_results/latent_extension/main/source-hashes.csv').open(encoding='utf-8-sig') as stream:
        recorded=[r['sha256'] for r in csv.DictReader(stream) if r['path']=='plugins/all-statements-slicing.jar']
    if not recorded or any(value!=jar for value in recorded): raise ValueError('旧切片工具版本不符')
    names=[f'slice.all_statements.{d}_slicing.log' for d in ('forward','backward')]
    for system in config['active_multifault']:
        base=Path(config['systems'][system])
        versions=sorted((base/f'4wise-{base.name}-1BUG-Full').glob('_MultipleBugs_*'))
        pairs=[]
        for v in versions:
            prior=(ROOT.parent/'email_trial/dataset'/v.name/'variants' if system=='email' else
                   Path('D:/dataset/whn/fp_extension_work/main')/system/v.name/'slices/variants')
            pairs.extend((variant,prior/variant.name) for variant in (v/'variants').iterdir())
        def one(pair):
            original,old=pair
            if not all((old/n).is_file() for n in names): return None
            raw,canonical=signatures(original)
            stamp=old/'inputs.sha256'
            if stamp.exists():
                if stamp.read_text()!=jar+' '+raw:return None
            elif not (old/'src').exists() or slice_inputs(old)!=raw:return None
            return dict(canonical=canonical,source=str(old),raw_input=raw,files={n:sha(old/n) for n in names})
        with ThreadPoolExecutor(max_workers=4) as pool:
            entries=[x for x in pool.map(one,pairs) if x is not None]
        by_key={}
        for item in entries:
            key=item['canonical']
            if key in by_key and by_key[key]['files']!=item['files']:
                for name in names:
                    left=json.loads((Path(by_key[key]['source'])/name).read_text())
                    right=json.loads((Path(item['source'])/name).read_text())
                    if {k:set(v) for k,v in left.items()}!={k:set(v) for k,v in right.items()}:
                        raise ValueError('相同输入的既有切片不同')
            by_key[key]=item
        target=base/'.fp_research/multifault-legacy-slices.json'
        atomic_json(target,dict(jar=jar,entries=by_key,validated_products=len(entries)))
        print(system,'legacy products',len(entries),'unique',len(by_key),flush=True)


@lru_cache(maxsize=3)
def load_index(base):
    target=base/'.fp_research/multifault-legacy-slices.json'
    if not target.exists(): return {}
    data=json.loads(target.read_text(encoding='utf8'))
    if data['jar']!=sha(ROOT.parent/'PULP/plugins/all-statements-slicing.jar'): raise ValueError('切片工具版本改变')
    return data['entries']


def seed(version):
    from . import case_study as study
    entries=load_index(version.parent.parent)
    if not entries:return 0
    cache=version.parent/'.fp_research/slice-cache'
    count=0
    for variant in sorted((version/'variants').iterdir()):
        _,canonical=signatures(variant)
        item=entries.get(canonical)
        if item is None:continue
        # 对已经实际运行过的新产品，额外核对完整前向、后向切片集合。
        existing=version/'.fp_research/multifault_joint/slices/variants'/variant.name
        for name,expected in item['files'].items():
            if (existing/name).exists() and sha(existing/name)!=expected:
                left=json.loads((existing/name).read_text())
                right=json.loads((Path(item['source'])/name).read_text())
                if {k:set(v) for k,v in left.items()}!={k:set(v) for k,v in right.items()}:
                    raise ValueError('新运行与既有切片集合不同，不能复用')
        with study._GUARD:
            lock=study._LOCKS.setdefault((str(cache),canonical),__import__('threading').Lock())
        with lock:
            key='legacy-'+canonical; target=cache/key
            if not (target/'complete.json').exists():
                target.mkdir(parents=True,exist_ok=True)
                for name,expected in item['files'].items():
                    source=Path(item['source'])/name
                    if sha(source)!=expected:raise ValueError('既有切片输出改变')
                    temp=target/(name+'.new')
                    temp.unlink(missing_ok=True)
                    try:os.link(source,temp)
                    except OSError:shutil.copyfile(source,temp)
                    os.replace(temp,target/name)
                atomic_json(target/'complete.json',dict(key=key,files=item['files'],backend='verified-legacy',
                    source=item['source'],input_sha256=item['raw_input'],canonical=canonical))
            with study._GUARD: study._ALIASES.setdefault(cache,{})[canonical]=key
            count+=1
    return count


if __name__=='__main__':index()
