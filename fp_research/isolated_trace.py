"""只对被选中的测试方法插桩，保留被测产品、测试脚手架及原始源码行号。"""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from .runtime import LIB,command
from .test_sources import methods
from .sites import site_of


def narrow_test(source,method):
    selected=[s for s in methods(source) if s[0]==method]
    if len(selected)!=1:
        raise ValueError("选中的测试方法不唯一")
    spans=list(methods(source))
    kept=source[selected[0][1]:selected[0][2]]
    for name,_,_ in spans:
        if name!=method and re.search(r"\b"+re.escape(name)+r"\s*\(",kept):
            return source
    for name,start,end in reversed(spans):
        if name!=method:
            source=source[:start]+re.sub(r"[^\r\n]"," ",source[start:end])+source[end:]
    return source


def single_trace(rt,variant,build,test,expected,stage,known,mapping):
    main,all_tests,classes=build
    cls,method=test.rsplit(".",1)
    short=cls.rsplit(".",1)[-1]+".java"
    sources=list((variant/"test").rglob(short))
    if len(sources)!=1:
        raise ValueError("测试类不唯一")
    stage.mkdir(parents=True,exist_ok=True)
    original=sources[0].read_text(encoding="utf-8-sig")
    source=stage/short
    source.write_text(narrow_test(original,method),encoding="utf8")
    selected=stage/"selected"
    selected.mkdir(exist_ok=True)
    classpath=os.pathsep.join([str(main),str(all_tests),LIB])
    command([rt.javac,"-encoding","UTF-8","-g","-cp",classpath,"-d",selected,source],stage)
    cp=os.pathsep.join([str(main),str(selected),str(all_tests),str(rt.agent),LIB])
    options="instrumentingclass="+":".join(sorted(set(classes+[cls])))+",logfile="+test+".log,project=fp"
    args=[rt.java,"-noverify","-javaagent:"+str(rt.agent)+"="+options,"-cp",cp,
          "org.junit.platform.console.ConsoleLauncher","--details=summary","--disable-ansi-colors",
          "--reports-dir",stage,"--select-method",cls+"#"+method]
    raw=stage/"stdout.log"
    with raw.open("wb") as out,(stage/"stderr.log").open("wb") as err:
        proc=subprocess.run(args,cwd=stage,stdout=out,stderr=err,timeout=150)
    cases=[case for report in stage.glob("TEST-*.xml") for case in ET.parse(report).iter("testcase")]
    if len(cases)!=1 or cases[0].get("classname")!=cls or cases[0].get("name","").split("(")[0]!=method:
        raise ValueError("测试运行器未精确执行所选方法")
    failure=cases[0].find("failure")
    if failure is None:
        failure=cases[0].find("error")
    outcome="failed" if failure is not None else "skipped" if cases[0].find("skipped") is not None else "passed"
    actual_site=site_of(failure,variant,mapping)
    if proc.returncode!=int(expected=="failed") or outcome!=expected:
        raise ValueError(f"测试记录 {expected}，插桩重放 {outcome}")
    if expected=="failed" and known is not None and known["site"] and actual_site!=known["site"]:
        raise ValueError(f"失效位置在插桩后改变 {known['site']} -> {actual_site}")
    trace=stage/"run.log"
    marker=("###"+cls+"::"+method).encode()
    found=False
    with raw.open("rb") as data,trace.open("wb") as dest:
        for line in data:
            if line.startswith((b"opcode",b"###")):
                dest.write(line)
                found |= line.strip()==marker
    if not found:
        raise ValueError("缺少目标测试的动态轨迹标记")
    return stage/"trace/logs/mytrace",trace
