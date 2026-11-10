"""仅从同一配置中选取记录为通过且观察目标一致的对照。"""
import argparse
from pathlib import Path

from .inventory import SYSTEMS, atomic_json, records
from .intents import overlap, version_intents


def pair_cases(cases, rows):
    pairs = {}
    for row in rows:
        prefix = row["configuration"] + "/"
        for failed in row["tests"]["failed"]:
            key = prefix + failed
            target = cases[key]
            choices = []
            for passed in row["tests"]["passed"]:
                other = cases[prefix + passed]
                shared = set(target["assertions"]) & set(other["assertions"])
                if not shared or set(target["operations"]) != set(other["operations"]):
                    continue
                same_class = failed.rpartition(".")[0] == passed.rpartition(".")[0]
                choices.append((not same_class, -overlap(target["assertions"], other["assertions"]),
                                -overlap(target["inputs"], other["inputs"]), passed, sorted(shared)))
            if choices:
                _, _, _, passed, shared = min(choices)
                pairs[key] = dict(control=prefix + passed, shared_observations=shared)
            else:
                pairs[key] = dict(control=None, reason="no_compatible_observation")
    return pairs


def prepare(root, system, version, output):
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("原始数据目录不能作为输出")
    for folder in sorted(root.joinpath(*SYSTEMS[system]).glob("_MultipleBugs_*")):
        if version and folder.name != version:
            continue
        rows = list(records(folder))
        cases, groups = version_intents(folder, rows)
        pairs = pair_cases(cases, rows)
        if sorted(pairs) != sorted(key for keys in groups.values() for key in keys):
            raise ValueError("配对与失效组成员不一致")
        atomic_json(output / system / folder.name / "semantics.json",
                    dict(cases=cases, groups=groups, pairs=pairs))
        print(system, folder.name, len(groups), len(pairs), sum(p["control"] is not None for p in pairs.values()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--system", choices=SYSTEMS)
    parser.add_argument("--version")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for system in [args.system] if args.system else SYSTEMS:
        prepare(args.data, system, args.version, args.output)
