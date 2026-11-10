from pathlib import Path

import networkx as nx
from dowhy import CausalModel


def parse_model_m(path):
    """
    解析 Elevator 的 model.m 文件，构建一个 DiGraph：
    - 节点：真正的特征（Base, Weight, Empty, ExecutiveFloor, TwoThirdsFull, Overloaded）
    - 节点属性：
        - mandatory: 是否必选（True = Base，False = 其他）
        - parent_model: 顶层模型名（这里是 "Elevator"）
    - 边：
        - 约束：A -> B，attr: kind='implies'
    图中 **不包含** Elevator 这个节点。
    """
    dot = nx.DiGraph()
    path = Path(path)
    nodes = []
    with path.open(encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("%")]

    # 1) 第一行：特征“树”的定义
    tree_line = lines[0]
    # 例： "Elevator : Base [Weight] [Empty] [ExecutiveFloor] [TwoThirdsFull] [Overloaded] :: _Elevator ;"
    tree_line = tree_line.rstrip(";")
    left, _right = tree_line.split("::", 1)
    root_part, children_part = left.split(":", 1)

    model_name = root_part.strip()           # 这里只当作模型名，不作为节点

    # children_part 里是子特征，空格分隔，可能带 []
    tokens = children_part.strip().split()

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue

        if tok.startswith("[") and tok.endswith("]"):
            child = tok[1:-1]          # 去掉中括号
            mandatory = False          # 可选特征
        else:
            child = tok
            mandatory = True           # 必选特征

        # 只加特征节点，不加 Elevator 节点，也不建 Elevator -> child 的边
        if child not in dot:
            dot.add_node(child, mandatory=mandatory, parent_model=model_name)
            nodes.append(child)
    # 2) 后续行：cross-tree constraints（implies）
    for line in lines[1:]:
        line = line.rstrip(";")
        if "implies" not in line:
            continue
        # 例： "TwoThirdsFull implies Weight"
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "implies":
            src = parts[0]
            dst = parts[2]
            # 确保节点存在
            if src not in dot:
                dot.add_node(src, mandatory=False, parent_model=model_name)
                nodes.append(src)
            if dst not in dot:
                dot.add_node(dst, mandatory=False, parent_model=model_name)
                nodes.append(src)
            dot.add_edge(dst, src)
    dot.add_node("__TEST_OUTPUT__")
    for feat in nodes:
        dot.add_edge(feat, "__TEST_OUTPUT__")
    return dot, nodes

try:
    import pandas as pd

    config_report_path = "/home/whn/codes/datasets/4wise-Elevator-FH-JML-1BUG-Full/_MultipleBugs_.NOB_1.ID_1/config.report.csv"
    # 读取完整的 config.report.csv，保留 Product\Feature 这一列
    df_full = pd.read_csv(config_report_path)

    # 复制一份用于数值化预处理：第一列是 Product\Feature，其余列是特征 + __TEST_OUTPUT__
    df_num = df_full.copy()

    # 从第二列开始都是需要数值化的列
    feature_cols = df_num.columns[1:]

    # 将 T/F 和 PASSED/FAILED（以及带空格版本）统一映射成 0/1
    replace_dict = {
        'T': 1,
        'F': 0,
        '  T  ': 1,
        '  F  ': 0,
        '__PASSED__': 1,
        '__FAILED__': 0,
    }
    df_num[feature_cols] = df_num[feature_cols].replace(replace_dict)

    # 强制转成 int，确保后续不会再有 object/字符串
    df_num[feature_cols] = df_num[feature_cols].astype(int)

    # 这里打印一下 df_num，确认已经是数值 + 保留 Product\Feature
    print("df:\n", df_num)
    m_path = "/home/whn/codes/datasets/4wise-Elevator-FH-JML-1BUG-Full/model.m"
    graph, nodes = parse_model_m(m_path)
    for node in nodes:
        model = CausalModel(data=df_num, treatment=node, outcome='__TEST_OUTPUT__', graph=graph)  # , graph=graphPath  graph
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(identified_estimand,
                                         method_name="backdoor.linear_regression")  # propensity_score_weighting  linear_regression
        print(node, estimate.value)
except:
    pass