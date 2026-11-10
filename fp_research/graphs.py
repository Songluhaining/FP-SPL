"""逐失效组重放已观察的测试并进行联合推理。输出仅保留可审计证据。"""
import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import tempfile
import subprocess

from .feature_sources import load_feature_map
from .inventory import SYSTEMS, atomic_json
from .runtime import ROOT, Runtime


def read_posterior(path):
    scores = {}
    for line in path.read_text(encoding="utf8").splitlines():
        fields = line.split("\t")
        if fields[0] != "STATEMENT":
            continue
        identity = base64.urlsafe_b64decode(fields[2] + "===").decode()
        correct = float(fields[3])
        if not math.isfinite(correct) or not 0 <= correct <= 1:
            raise ValueError("无效的语句后验值")
        if not identity.startswith("unmapped:"):
            scores[identity] = 1 - correct
    return scores


def process_version(job):
    version, out, cache, java, limit = job
    runtime = Runtime(java, cache)
    semantics = json.loads((out / "semantics.json").read_text(encoding="utf8"))
    groups, pairs = semantics["groups"], semantics["pairs"]
    with tempfile.TemporaryDirectory(prefix="runs-", dir=out) as tmp:
        work = Path(tmp).resolve()
        builds, maps, executions = {}, {}, {}
        for group, members in list(groups.items())[:limit]:
            identifiers = sorted(set(members) | {pairs[k]["control"] for k in members if pairs[k]["control"]})
            run_spec = {key:"failed" if key in members else "passed" for key in identifiers}
            # 绑定本组源码、覆盖映射、执行参数、分析代码及代理源码。原始数据始终只读。
            digest = hashlib.sha256(json.dumps(run_spec, sort_keys=True).encode())
            for p in [Path(__file__), ROOT/"fp_research/runtime.py", ROOT/"fp_research/feature_sources.py", cache/"source.sha256"]:
                digest.update(p.read_bytes())
            manifests = []
            for config in sorted({key.split("/")[0] for key in identifiers}):
                variant = version/"variants"/config
                for sub in ["src","test"]:
                    for p in sorted((variant/sub).rglob("*.java")):
                        digest.update(p.read_bytes())
                for p in sorted((variant/"coverage").glob("spectrum_*.xml")):
                    digest.update(p.read_bytes())
            output = out/"graphs"/(group+".json")
            key_hash = digest.hexdigest()
            if output.exists():
                old = json.loads(output.read_text(encoding="utf8"))
                if old["fingerprint"] == key_hash and old["status"] == "ok":
                    continue
            record = dict(fingerprint=key_hash, executions=run_spec)
            try:
                for identity, expected in run_spec.items():
                    config, test = identity.split("/")
                    variant = version/"variants"/config
                    if config not in builds:
                        builds[config] = runtime.compile(variant,work/config)
                        maps[config] = work/config/"map.tsv"
                        mapping = load_feature_map(variant)
                        maps[config].write_text("".join(f"{cls}#{line}\t{value}\n" for (cls,line),value in sorted(mapping.items())),encoding="utf8")
                    if identity not in executions:
                        stage = work/config/hashlib.sha256(test.encode()).hexdigest()[:12]
                        executions[identity] = runtime.trace(builds[config],test,stage,expected)
                    sources, trace = executions[identity]
                    manifests.append((identity,sources,trace,test,"PASS" if expected=="passed" else "FAIL",maps[config]))
                with tempfile.TemporaryDirectory(dir=work) as scratch:
                    scores = read_posterior(runtime.infer(Path(scratch), manifests))
                record.update(status="ok" if scores else "unmapped", scores=scores)
            except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
                record.update(status="error",reason=str(error)[-2500:])
            atomic_json(output,record)
            if limit:
                print(version.name,group,record["status"],record.get("reason","")[-350:],flush=True)
    print("finished",version.name,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--java-home",type=Path,required=True)
    parser.add_argument("--system",choices=SYSTEMS)
    parser.add_argument("--version")
    parser.add_argument("--limit-groups",type=int)
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.is_relative_to(args.data.resolve()):
        parser.error("工作目录必须在数据目录外")
    cache=out/"runtime"
    Runtime(args.java_home,cache).build()
    jobs=[]
    for system in [args.system] if args.system else SYSTEMS:
        for version in sorted(args.data.joinpath(*SYSTEMS[system]).glob("_MultipleBugs_*")):
            if not args.version or args.version==version.name:
                jobs.append((version.resolve(),out/system/version.name,cache,args.java_home,args.limit_groups))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(process_version,jobs))
