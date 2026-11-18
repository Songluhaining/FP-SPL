"""用最小击中集求解原版SPC条件，保留至多7个特征选择的限制。"""
from pathlib import Path


def minimal_sets(collection):
    ordered = sorted(set(map(frozenset, collection)), key=lambda x: (len(x), sorted(x)))
    kept = []
    for candidate in ordered:
        if not any(previous <= candidate for previous in kept):
            kept.append(candidate)
    return kept


def minimal_hitting_sets(switches, limit=7):
    """每个通过配置至少有一项特征选择与当前失败配置不同。"""
    switches = minimal_sets(switches)
    if any(not switch for switch in switches):
        return []
    choices = [frozenset()]
    for switch in switches:
        expanded = []
        for candidate in choices:
            if candidate & switch:
                expanded.append(candidate)
            elif len(candidate) < limit:
                expanded.extend(candidate | {literal} for literal in switch)
        choices = minimal_sets(expanded)
        if not choices:
            break
    return choices


def detect(feature_names, passed, failed, names, variants_dir, target):
    """输出与原版 detect_SPCs 相同的(特征选择组合,失败产品源码路径)集合。"""
    if not passed or not failed:
        Path(target).write_text("", encoding="utf-8")
        return str(target)
    from spc import SPCsManager as original
    # 保留原始特征集合的迭代次序，避免改变Java端按序处理特征的行为。
    output = []
    found = []
    for failure in failed:
        switches = [original.find_switched_feature_selections(failure, passing) for passing in passed]
        switches = original.minimize_switches(switches)
        order = list(original.union_all_switched_feature_selections(switches))
        candidates = minimal_hitting_sets(switches)
        candidates.sort(key=lambda c: (len(c), tuple(i for i, literal in enumerate(order) if literal in c)))
        for selected in candidates:
            candidate = set(literal for literal in order if literal in selected)
            if original.satisfy_spc_minimality(candidate, found):
                combined = original.combine_spc_with_feature_names(feature_names, candidate)
                for matched in original.find_failed_configs_contains_spc(candidate, failed):
                    source = Path(variants_dir) / names[tuple(matched)] / "src"
                    output.append(f"{combined}; {source.as_posix()}\n")
                found.append(candidate)
    Path(target).write_text("".join(output), encoding="utf-8")
    return str(target)
