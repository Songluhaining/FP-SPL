"""按系统和缺陷版本划分实验，不使用真假通过标签。"""
import hashlib


def splits(root, systems=("bank", "email")):
    first, second = systems
    versions={s:sorted(p for p in (root/s).glob("_MultipleBugs_*") if (p/"inventory.json").exists()) for s in systems}
    for s,paths in versions.items():
        order=sorted(paths,key=lambda p:hashlib.sha256(("42/"+s+"/"+p.name).encode()).digest())
        for fold in range(5):
            test=order[fold::5]
            yield f"{s}-fold{fold}",[p for p in order if p not in test],test
        other=second if s==first else first
        yield f"{other}-to-{s}",versions[other],paths
