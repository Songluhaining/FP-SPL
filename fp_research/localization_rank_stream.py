"""各案例的三份折外预测就绪后立即定位，引用不可变的单折文件。"""
from concurrent.futures import ProcessPoolExecutor
import os
import time

from .localization_study import OUTPUT, manifest, perform, read
from .varcop_localization import baseline_path, source_hashes
from .inventory import atomic_json


def main():
    os.environ["PYTHONHASHSEED"] = "42"
    jobs = manifest()
    while not (OUTPUT / "adapters-verified.json").exists():
        time.sleep(10)
    if read(OUTPUT / "adapters-verified.json")["source"] != source_hashes():
        raise ValueError("当前适配源码尚未通过代表案例核验")
    waiting = list(jobs)
    failures, finished = 0, 0
    with ProcessPoolExecutor(max_workers=6) as pool:
        running = {}
        while waiting or running:
            ready = [j for j in waiting if all(baseline_path(j, m).with_suffix(".json").exists()
                       for m in ("CLAP", "PULP", "PULP-default"))]
            for job in ready[:max(0, 6-len(running))]:
                running[pool.submit(perform, job, "rank")] = job
                waiting.remove(job)
            for future in list(running):
                if future.done():
                    failures += not future.result()
                    finished += 1
                    del running[future]
                    print("完成定位", finished, "/", len(jobs), "失败", failures, flush=True)
            time.sleep(2)
    if failures:
        raise SystemExit(f"{failures} cases failed; inspect rank-status.json")
    atomic_json(OUTPUT / "rank-stream-complete.json", dict(cases=finished, failures=failures))


if __name__ == "__main__":
    main()
