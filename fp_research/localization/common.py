"""VARCOP固定快照和只读覆盖适配，不包含产品剔除。"""
from functools import lru_cache
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pandas as pd
from ..config import ROOT,tool
from ..io import read,digest,atomic_json
from ..experiments.catalog import METRICS,UPSTREAM,correction_sets,historical_work_folder as work_folder
JAVA=tool('java_home')/'bin/java.exe'
JAR=tool('varcop_jar')
VENDOR=ROOT/'fp_research/vendor/varcop'
def source_hashes():
    paths=[*VENDOR.rglob('*.py'),*(ROOT/'fp_research/localization').glob('*.py'),ROOT/'tools/java/VarcopFeatureProfile.java',
           ROOT/'configs/tools.json',ROOT/'fp_research/experiments/catalog.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}
def upstream():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from util import FileManager
    FileManager.join_path = lambda *p: Path(os.path.join(*p)).as_posix()
    from ranking import RankingManager as ranking
    from ranking import VariantLevelRankingManager as products
    from spc import SPCsManager
    logging.getLogger().setLevel(logging.WARNING)
    return ranking, products, SPCsManager

def input_fingerprint(version):
    paths = [version / "config.report.csv.done"]
    for variant in sorted((version / "variants").iterdir()):
        paths.extend(sorted((variant / "coverage").glob("spectrum_*_coverage.xml")))
        paths.extend(sorted((variant / "src").rglob("*.java")))
    hashes = {str(p.relative_to(version)): digest(p) for p in paths}
    key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return key, hashes

def read_isolation(version, sliced, excluded, names):
    raw = read(sliced)
    data = {}
    for cid, statements in raw.items():
        if cid not in names or cid in excluded:
            raise ValueError("特征切片出现不可用配置")
        failed_path = version / "variants" / cid / "coverage/spectrum_failed_coverage.xml"
        executed = {line.get("featureClass")+"."+line.get("featureLineNum")
                    for line in ET.parse(failed_path).iterfind(".//package/file/line")
                    if int(line.get("count", 0)) > 0}
        data[cid] = {key: value for key, value in statements.items() if key in executed}
    return data
