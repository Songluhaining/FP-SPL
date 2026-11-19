"""按系统和缺陷层固定五折，每个版本整体留出一次。"""
import hashlib


def partitions(jobs,folders):
    identities={folders[(j['system'],j['version'])]:(j['system'],j['version']) for j in jobs}
    parts=[]
    for system,stratum in sorted({(j['system'],j['stratum']) for j in jobs}):
        chosen=[j for j in jobs if (j['system'],j['stratum'])==(system,stratum)]
        chosen.sort(key=lambda j:hashlib.sha256(('42/'+system+'/'+j['version']).encode()).digest())
        ordered=[folders[(j['system'],j['version'])] for j in chosen]
        if len(ordered)<2: raise ValueError('分类比较至少需要同系统同缺陷层两个版本，不能用单案例训练自身')
        for fold in range(5):
            testing=ordered[fold::5]
            if testing: parts.append((f'{system}-{stratum}fault-fold{fold}',[p for p in ordered if p not in testing],testing))
    return parts,identities
