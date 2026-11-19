"""以源码和实际执行语句登记切片等价输入。"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
def signatures(variant):
    source=hashlib.sha256(); raw=hashlib.sha256(); executed=set()
    for p in sorted((variant/"src").rglob("*.java")):
        item=p.relative_to(variant).as_posix().encode()+b"\0"+hashlib.sha256(p.read_bytes()).digest()
        source.update(item); raw.update(item)
    for p in sorted((variant/"coverage").glob("spectrum_*_coverage.xml")):
        payload=p.read_bytes()
        raw.update(p.relative_to(variant).as_posix().encode()+b"\0"+hashlib.sha256(payload).digest())
        if p.name in {"spectrum_failed_coverage.xml","spectrum_passed_coverage.xml"}:
            for line in ET.fromstring(payload).iter("line"):
                if "count" in line.attrib and int(line.get("count"))>0:
                    executed.add(line.get("featureClass","")+"."+line.get("featureLineNum",""))
    canonical=source.hexdigest()+json.dumps(sorted(executed),ensure_ascii=True,separators=(",",":"))
    return raw.hexdigest(),hashlib.sha256(canonical.encode()).hexdigest()
