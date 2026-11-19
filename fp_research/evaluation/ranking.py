from pathlib import Path
"""官方最坏并列排名及候选范围回退规则。"""
def locate(statement, ranked):
    """保持官方次序与最坏的连续(score,secondary)并列，独立于官方查询函数。"""
    for start, row in enumerate(ranked):
        if row[0] == statement:
            end = start
            while end+1 < len(ranked) and tuple(ranked[end][1:]) == tuple(ranked[end+1][1:]):
                end += 1
            return end+1
    return None

def fault_positions(version):
    result = set()
    for line in (version / (version.name + ".mutant.log")).read_text(encoding="utf-8").splitlines():
        location, number, _ = line.split(":", 2)
        result.add(location.rsplit(".", 1)[0] + "." + number)
    return sorted(result)

def layered_position(statement, full, focus):
    found = locate(statement, focus)
    if found is not None:
        return found
    position = locate(statement, full)
    if position is None:
        return None
    isolated = {r[0] for r in focus}
    return len(focus) + position - sum(r[0] in isolated for r in full[:position])
def assert_same_ranking(expected, actual):
    """并列语句的文件顺序可变；核对每条分数、并列层次及最坏名次。"""
    expected = [list(row) for row in expected]
    actual = [list(row) for row in actual]
    assert len(expected) == len(actual)
    assert {r[0]: r[1:] for r in expected} == {r[0]: r[1:] for r in actual}
    assert [r[1:] for r in expected] == [r[1:] for r in actual]
    assert {r[0]: locate(r[0], expected) for r in expected} == {
        r[0]: locate(r[0], actual) for r in actual}
