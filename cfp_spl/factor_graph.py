from __future__ import annotations

import base64
import re
from collections import defaultdict, deque
from pathlib import Path

from .domain import FactorGraph, StatementPosterior


LINE_INFO = re.compile(r"^(?P<head>.+)#(?P<line>\d+)#(?P<offset>-?\d+)(?:#.*)?$")


def decode(value: str) -> str:
    if not value:
        return ""
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", errors="replace")


def stable_statement_key(raw_name: str) -> str:
    """移除测试名和动态计数，保留类、源码行和字节码位置。"""
    match = LINE_INFO.match(raw_name)
    if match:
        head = match.group("head").replace("/", ".")
        return f"{head}#{match.group('line')}#{match.group('offset')}"
    return raw_name.replace("/", ".")


def load_factor_graph(path: Path) -> FactorGraph:
    graph = FactorGraph()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        kind = fields[0]
        try:
            if kind == "STATEMENT":
                raw = decode(fields[2])
                graph.statements[fields[1]] = StatementPosterior(
                    fields[1], raw, stable_statement_key(raw), float(fields[3]),
                    fields[4] == "1", fields[5] == "1",
                )
            elif kind == "VALUE":
                graph.values[fields[1]] = {
                    "name": decode(fields[2]), "origin": decode(fields[3]),
                    "stack_values": decode(fields[4]).split("|") if fields[4] else [],
                    "correctness": float(fields[5]), "observed": fields[6] == "1",
                }
            elif kind == "FACTOR":
                graph.factors[fields[1]] = {
                    "statement": decode(fields[2]), "operators": decode(fields[3]).split("|"),
                    "tensor": [float(x) for x in decode(fields[4]).split(",") if x],
                }
            elif kind == "INCIDENCE":
                graph.incidences.append((fields[1], fields[2], fields[3]))
        except (IndexError, ValueError) as exc:
            raise ValueError(f"因子图第{number}行格式错误：{line}") from exc
    return graph


def causal_statement_keys(graph: FactorGraph) -> set[str]:
    """从被观测值沿二部因子图反向寻找能影响结果的已执行语句。"""
    factor_to_nodes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    node_to_factors: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for factor, node, role in graph.incidences:
        factor_to_nodes[factor].append((node, role))
        node_to_factors[node].append((factor, role))
    observed = [node for node, value in graph.values.items() if value.get("observed")]
    observed += [node for node, stmt in graph.statements.items() if stmt.observed]
    queue = deque(observed)
    seen_nodes, seen_factors = set(observed), set()
    while queue:
        identity = queue.popleft()
        for factor, role_at_node in node_to_factors.get(identity, []):
            # BRFL因子由STMT/PRED/USE共同产生DEF。只从结果端反向进入因子，
            # 防止沿某个值的所有消费者向前扩张成整个连通分量。
            if role_at_node != "DEF" and not (identity in graph.statements and role_at_node == "STMT"):
                continue
            if factor in seen_factors:
                continue
            seen_factors.add(factor)
            for node, role in factor_to_nodes.get(factor, []):
                if role == "DEF":
                    continue
                if node not in seen_nodes:
                    seen_nodes.add(node)
                    queue.append(node)
    return {graph.statements[node].stable_key for node in seen_nodes if node in graph.statements}


def statement_index(graph: FactorGraph) -> dict[str, StatementPosterior]:
    index: dict[str, StatementPosterior] = {}
    for stmt in graph.statements.values():
        previous = index.get(stmt.stable_key)
        if previous is None or stmt.suspiciousness > previous.suspiciousness:
            index[stmt.stable_key] = stmt
    return index
