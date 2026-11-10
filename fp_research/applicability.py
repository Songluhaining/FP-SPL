"""按构建映射和启用特征汇总适用分数，单独记录未执行的代码。"""
import xml.etree.ElementTree as ET


def build_evidence(folder):
    present, executed = set(), set()
    for report in sorted((folder / "coverage").glob("spectrum_*_coverage.xml")):
        for line in ET.parse(report).iterfind(".//package/file/line"):
            feature, num = line.get("featureClass"), line.get("featureLineNum")
            if feature is None or num is None:
                continue
            key = f"{feature}:{num}"
            present.add(key)
            if any(int(line.get(attr, 0)) > 0 for attr in ("count", "truecount", "falsecount")):
                executed.add(key)
    return present, executed


def applicability(scores, selected, present, executed):
    if not present or not scores:
        return None
    weight = sum(scores.values())
    if weight <= 0:
        return None
    matched = {k for k in scores if k in present and k.split(".",1)[0] in selected}
    return dict(score=sum(scores[k] for k in sorted(matched)) / weight,
                included=len(matched), executed=len(matched & executed),
                present_unexecuted=len(matched - executed),
                leading=sorted(matched,key=lambda k:(-scores[k],k))[:5])
