"""从测试语法树解析被测操作、断言目标和输入，按可观测行为分组。"""
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import javalang
from javalang import tree


def expression_targets(node, bindings):
    if node is None:
        return set()
    targets = set()
    for _, part in node:
        if isinstance(part, tree.MethodInvocation):
            targets.add("call:" + part.member)
        elif isinstance(part, tree.MemberReference):
            targets.update(bindings.get(part.member, {"field:" + part.member}))
        elif isinstance(part, tree.ClassCreator):
            targets.add("new:" + part.type.name)
    return targets


def describe(method):
    bindings = {}
    calls, observed, inputs = set(), set(), set()
    for _, node in method:
        if isinstance(node, tree.VariableDeclarator):
            bindings[node.name] = expression_targets(node.initializer, bindings)
        elif isinstance(node, tree.Assignment) and isinstance(node.expressionl, tree.MemberReference):
            if not node.expressionl.qualifier:
                bindings[node.expressionl.member] = expression_targets(node.value, bindings)
        elif isinstance(node, tree.MethodInvocation):
            if node.member.startswith("assert"):
                # 最后一个参数有时是数值比较容差，因此保留所有有来源的参数。
                for arg in node.arguments:
                    for target in expression_targets(arg, bindings):
                        observed.add(node.member + "/" + target)
            elif node.member not in {"fail", "verifyException"}:
                calls.add(node.member)
        elif isinstance(node, tree.CatchClauseParameter):
            observed.update("exception/" + name for name in node.types)
        elif isinstance(node, tree.Literal):
            inputs.add("".join(node.prefix_operators or []) + node.value)
    return dict(operations=sorted(calls), assertions=sorted(observed), inputs=sorted(inputs))


@lru_cache(maxsize=256)
def parse_source(content):
    unit = javalang.parse.parse(content)
    return {m.name: describe(m) for _, m in unit.filter(tree.MethodDeclaration)
            if any(a.name.split(".")[-1] == "Test" for a in m.annotations)}


def case_descriptions(folder):
    cases = {}
    for path in sorted((folder / "test").rglob("*Test.java")):
        source = path.read_text(encoding="utf-8-sig", errors="replace")
        package = re.search(r"\bpackage\s+([\w.]+)\s*;", source)
        prefix = ((package.group(1) + ".") if package else "") + path.stem + "."
        for name, desc in parse_source(source).items():
            cases[prefix + name] = desc
    return cases


def executed_lines(xml_path):
    lines = set()
    for file in ET.parse(xml_path).iterfind(".//package/file"):
        for line in file.findall("line"):
            if line.get("type") == "stmt" and int(line.get("count", 0)) > 0:
                lines.add((file.get("name"), int(line.get("num"))))
    return lines


def overlap(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left or right else 0.0


def group_failures(failures):
    groups = defaultdict(list)
    for identity, desc, trace in sorted(failures):
        key = (tuple(desc["assertions"]), tuple(desc["operations"]))
        # 空判据的用例不依靠彼此相似的名字强行合并。
        if not any(key):
            key = (identity,)
        clusters = groups[key]
        cluster = next((c for c in clusters if overlap(trace, c[0][1]) >= 0.7), None)
        if cluster is None:
            cluster = []
            clusters.append(cluster)
        cluster.append((identity, trace))
    result = {}
    for clusters in groups.values():
        for cluster in clusters:
            ids = [i for i, _ in cluster]
            group = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
            result[group] = ids
    if sum(map(len, result.values())) != len(failures):
        raise ValueError("失效组成员总数不一致")
    return result


def version_intents(version, rows):
    cases, failures = {}, []
    for row in rows:
        folder = version / "variants" / row["configuration"]
        descriptions = case_descriptions(folder)
        for outcome, tests in row["tests"].items():
            for name in tests:
                key = row["configuration"] + "/" + name
                cases[key] = descriptions[name]
                if outcome == "failed":
                    path = folder / "coverage" / "failed" / (name + ".coverage.xml")
                    failures.append((key, descriptions[name], executed_lines(path)))
    return cases, group_failures(failures)
