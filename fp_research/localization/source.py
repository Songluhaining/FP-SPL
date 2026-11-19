"""按固定提交取得官方特征切片程序，并核对仓库内的源码快照。"""
import hashlib
import json
from pathlib import Path
import subprocess

from ..config import ROOT,tool
from ..experiments.catalog import UPSTREAM


def main():
    metadata = json.loads((ROOT / "fp_research/vendor/varcop/SOURCE.json").read_text(encoding="utf-8"))
    for name, sha in metadata["files"].items():
        source = ROOT / "fp_research/vendor/varcop" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != sha:
            raise ValueError("官方源码快照改变 " + name)
    checkout = ROOT / "result/varcop_source"
    if not (checkout / ".git").exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/ttrangnguyen/VARCOP.git", str(checkout)], check=True)
    command = ["git", "-c", "safe.directory="+checkout.as_posix(), "-C", str(checkout)]
    if subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip() != UPSTREAM:
        subprocess.run([*command, "fetch", "--depth", "1", "origin", UPSTREAM], check=True)
    jar = tool('varcop_jar')
    jar.parent.mkdir(parents=True,exist_ok=True)
    if not jar.exists() or hashlib.sha256(jar.read_bytes()).hexdigest() != metadata["feature_slicing_jar_sha256"]:
        data = subprocess.check_output([*command, "show", UPSTREAM+":plugins/feature-slicing.jar"])
        if hashlib.sha256(data).hexdigest() != metadata["feature_slicing_jar_sha256"]:
            raise ValueError("官方插件摘要不符")
        jar.write_bytes(data)
    print(UPSTREAM, jar, "verified", flush=True)


if __name__ == "__main__":
    main()
