"""统一解析仓库路径、数据和外部工具。"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LOCATIONS=ROOT/'configs/datasets.json'
def datasets(): return json.loads(LOCATIONS.read_text(encoding='utf-8'))
def tool(name):
    value=Path(json.loads((ROOT/'configs/tools.json').read_text(encoding='utf-8'))[name])
    return value if value.is_absolute() else (ROOT/value).resolve()
