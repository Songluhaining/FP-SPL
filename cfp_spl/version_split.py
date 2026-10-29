from __future__ import annotations

import hashlib


def fold_of(version: str, fold_count: int = 5) -> int:
    """按版本名稳定分折，不受某个阶段缺少部分版本的影响。"""
    if fold_count < 2:
        raise ValueError("折数至少为二")
    digest = hashlib.sha256(version.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % fold_count


def split_versions(versions: set[str] | list[str] | tuple[str, ...],
                   validation_fold: int = 0, fold_count: int = 5
                   ) -> tuple[set[str], set[str]]:
    if not 0 <= validation_fold < fold_count:
        raise ValueError("验证折编号必须小于折数")
    values = set(versions)
    validation = {name for name in values
                  if fold_of(name, fold_count) == validation_fold}
    return values - validation, validation
