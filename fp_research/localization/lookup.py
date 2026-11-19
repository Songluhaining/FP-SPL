"""为官方只读局部排名建立索引，保持首次匹配和缺失值返回完全一致。"""
from contextlib import contextmanager


@contextmanager
def indexed_lookup(ranking):
    original = ranking.get_local_score
    # 保留原列表引用，防止列表释放后的标识复用；每个案例结束即清空。
    tables = {}

    def lookup(statement, sequence):
        key = id(sequence)
        if key not in tables:
            values = {}
            for row in sequence:
                values.setdefault(row[0], (row[1], row[2]))
            tables[key] = (sequence, values)
            # 每份新索引逐项与原查找比较；不改变任何分数或列表次序。
            for name, expected in values.items():
                if original(name, sequence) != expected:
                    raise ValueError("索引查找与原始查找不一致")
        return tables[key][1].get(statement, (ranking.STM_NOT_FOUND, 0))

    ranking.get_local_score = lookup
    try:
        yield
    finally:
        ranking.get_local_score = original
