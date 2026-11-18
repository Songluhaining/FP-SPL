"""核对银行账户已有单缺陷切片，为多缺陷基线生成输入索引。"""
from concurrent.futures import ThreadPoolExecutor
import json
import msvcrt
import os
from pathlib import Path
import time

from .case_study import ROOT
from .inventory import atomic_json
from .multifault_cache import sha
from .slice_reuse import signatures
from .extension import slice_inputs
from .multifault_cache import seed as original_seed


def seed_with_process_lock(version, raw_digest=None):
    cache = version.parent / ".fp_research/slice-cache"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "localization-seed.lock").open("a+b") as stream:
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(.1)
        try:
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"0")
                stream.flush()
            return original_seed(version, raw_digest)
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def index_bank():
    base = Path("D:/dataset/whn/BankAccountTP")
    names = [f"slice.all_statements.{direction}_slicing.log" for direction in ("forward", "backward")]
    jar = sha(ROOT.parent / "PULP/plugins/all-statements-slicing.jar")
    pairs = []
    for version in sorted((base / "4wise-BankAccountTP-1BUG-Full").glob("_MultipleBugs_*")):
        old = ROOT.parent / "pulp_trial/dataset" / version.name / "variants"
        pairs.extend((v, old / v.name) for v in (version / "variants").iterdir())

    def checked(pair):
        original, old = pair
        if not all((old / name).is_file() for name in names):
            return None
        raw, canonical = signatures(original)
        stamp = old / "inputs.sha256"
        if stamp.exists():
            if stamp.read_text() != jar+" "+raw:
                return None
        elif not (old / "src").exists() or slice_inputs(old) != raw:
            return None
        return dict(canonical=canonical, source=str(old), raw_input=raw,
                    files={name: sha(old / name) for name in names})

    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = [item for item in pool.map(checked, pairs) if item is not None]
    unique = {}
    for item in entries:
        key = item["canonical"]
        if key in unique and unique[key]["files"] != item["files"]:
            for name in names:
                left = json.loads((Path(unique[key]["source"]) / name).read_text())
                right = json.loads((Path(item["source"]) / name).read_text())
                if {k: set(v) for k, v in left.items()} != {k: set(v) for k, v in right.items()}:
                    raise ValueError("银行账户相同输入的已有切片不一致")
        unique[key] = item
    atomic_json(base / ".fp_research/multifault-legacy-slices.json",
                dict(jar=jar, entries=unique, validated_products=len(entries)))
    print("bank", len(entries), "verified products", len(unique), "distinct inputs", flush=True)


if __name__ == "__main__":
    index_bank()
