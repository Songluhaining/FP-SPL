"""提前计算原始产品的静态关系，不读取识别结果或评价标签。"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import os
from pathlib import Path
import time

from .localization_study import manifest, work_folder


def warm(job):
    from . import varcop_localization as v
    from .varcop_slice import batch
    started = time.monotonic()
    version = Path(job["path"])
    folder = work_folder(job) / "static-preparation"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with (version / "config.report.csv.done").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header, rows = next(reader), list(reader)
        _, _, spc = v.upstream()
        log = v.write_spcs(version, folder, set(), rows, header, spc)
        batch(version, {"original": (log, folder / "slicing.json")}, folder)
        v.atomic_json(folder / "complete.json", dict(seconds=time.monotonic()-started))
        print(job["system"], job["version"], "静态关系就绪", flush=True)
        return True
    except Exception as error:
        v.atomic_json(folder / "error.json", dict(type=type(error).__name__, reason=str(error)))
        print(job["system"], job["version"], type(error).__name__, str(error), flush=True)
        return False


def main():
    os.environ["PYTHONHASHSEED"] = "42"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        failures = sum(not future.result() for future in as_completed(
            [pool.submit(warm, job) for job in manifest()]))
    print("静态准备结束，失败数", failures, flush=True)


if __name__ == "__main__":
    main()
