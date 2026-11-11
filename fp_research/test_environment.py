"""只在隔离编译副本中映射自动生成测试的环境目录，保留测试及产品原文。"""
import hashlib
import json
import os
from pathlib import Path
import re

from .runtime import LIB, command


PROPERTIES=re.compile(r'(System\.setProperty\("(user\.dir|user\.home|java\.io\.tmpdir)",\s*)"([^"]*)"')


def mapped_scaffolding(body, working):
    changes=[]
    def replace(match):
        changes.append(dict(property=match[2],original=match[3],mapped=str(working)))
        return match[1]+json.dumps(str(working),ensure_ascii=True)
    return PROPERTIES.sub(replace,body),changes


def recompile_scaffolding(rt,variant,working,main,tests):
    folder=working/'portable-scaffolding'
    folder.mkdir(parents=True,exist_ok=True)
    inputs=[]
    receipt=[]
    for source in sorted((variant/'test').rglob('*_scaffolding.java')):
        body=source.read_text(encoding='utf-8-sig')
        mapped,changes=mapped_scaffolding(body,working.resolve())
        if not changes:continue
        target=folder/source.relative_to(variant/'test')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(mapped,encoding='utf8')
        inputs.append(target)
        receipt.append(dict(source=str(source.relative_to(variant)),changes=changes,
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            compiled_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    if not inputs:raise ValueError('未发现可映射的测试环境目录')
    command([rt.javac,'-g','-encoding','UTF-8','-cp',os.pathsep.join([str(main),str(tests),LIB]),
             '-d',tests,*inputs],working)
    return receipt
