# FP-SPL：基于逐测试动态依赖图的假通过配置排序

当前主流程使用已移入本仓库的 BRFL 字节码插桩和运行因子图代码，不读取 PULP 的切片日志，也不需要外部 BRFL 仓库。输入中的 `__FAILED__` 只作为已知故障行为样本，`__PASSED__` 只作为无标签候选；真假通过研究标签不进入训练。

## 实现流程

1. 从本仓库 `vendor/brfl/src/main/java` 编译插桩代理和图构建库；原 BRFL 仓库保持不变。
2. 将每个配置的产品源码与测试源码以 `javac -g` 重新编译到输出目录，保留字节码到源码的行号映射。原始数据集保持只读。
3. 对每条测试分别启动 Java 虚拟机，使用 BRFL 代理记录 `opcode` 轨迹与 `.source.log`；核对实际测试结果与覆盖报告。
4. 使用 BRFL 的 `ByteCodeGraph` 对单条测试的源码日志和运行日志构图。导出每个运行因子的定义、使用与条件前驱，形成有方向且标注 `DATA`／`CONTROL` 的直接依赖边；保留事件数量和运行值。
5. 按覆盖文件中的 `featureClass`、`featureLineNum` 把源码位置映射至特征来源。以测试断言为可观察点，对依赖图取后向闭包，得到失败测试的局部行为模板和通过测试的局部图。基础特征也正常保留。
6. 默认使用三层关系感知图神经网络：以代码位置、特征来源、代码行、分支结果和数值摘要作为节点属性；沿有向的数据／控制依赖边及其反向边传递消息，以注意力汇总逐测试局部图。失败测试作为正样本，通过测试作为可能混有假通过的低权重背景，训练 4 个不同背景抽样的网络。候选测试优先使用未抽取其所属配置的网络评分；配置分数取其通过测试的最高分。失败模板的图相似度只作为解释证据，不参与默认图模型分数。

此分数是**疑似假通过排序分数**，并非校准概率或最终真假判定。节点的源码表示目前是可训练的散列嵌入，尚未接入预训练代码语义模型或因果干预。样本较少时，深度网络可能过拟合；应按缺陷版本进行独立验证，并与 `--model linear` 的原线性图特征模型比较。

## 运行条件

- Python 3.10 及以上，默认图模型另需安装 `torch`（例如 `pip install -e ".[graph]"`）。原线性模型需要 `numpy`、`scikit-learn`。
- Java 8 开发工具包。插桩源码和运行所需依赖包随本仓库存放在 `vendor/brfl/`。
- BankAccountTP 原始数据集有 `src/`、`test/`、逐测试覆盖及 `config.report.csv.done`。

```powershell
cd E:\Myself\投稿\work7\FP-SPL
python -m fpspl --method dynamic `
  --dataset D:\dataset\whn\4wise-BankAccountTP-1BUG-Full `
  --java-home 'E:\Program Files\Java\jdk-1.8' `
  --output E:\Myself\投稿\work7\FP-SPL\outputs\dynamic-bankaccount `
  --version _MultipleBugs_.NOB_1.ID_98
```

去掉 `--version` 会依次处理全部缺陷版本。初次检查可用 `--limit-failed-variants 2 --limit-passing-variants 2 --max-tests-per-variant 2`；这些限制会改变训练样本和排序范围，不能当作完整实验结果。

已有 `test_graphs.jsonl.gz` 时，可直接重新训练图模型，避免再次插桩和运行测试：

```powershell
python -m fpspl.retrain_graph `
  --graphs outputs\dynamic-eval194\_MultipleBugs_.NOB_1.ID_194\test_graphs.jsonl.gz `
  --output outputs\graph-eval194
```

## 输出与核查

- `<版本>/instrumented/<配置>/build/`：在工作区重编译的字节码。
- `<版本>/instrumented/<配置>/tests/<测试>/`：单条测试的代理轨迹、构图结果 `graph.tsv` 和实际结果 `status.json`。
- `<版本>/test_graphs.jsonl.gz`：逐测试的局部动态图；`typed_links` 是带方向和类型的直接运行依赖边。
- `<版本>/failure_templates.json`：失败测试的局部故障行为模板。
- `<版本>/graph_model.pt`：默认图神经网络的权重；选择 `--model linear` 时输出 `pu_model.pkl`。只应加载可信的本地模型文件。
- `<版本>/ranking.csv`：原本通过的配置及证据排序。
- `<版本>/summary.json`：图数量、是否完成训练、未能重放或解析的测试。

训练和排序不读取真假通过研究标签。若另有标签，请在排序完成后另行评估；按缺陷版本划分，分别报告原本通过和由失败配置转换的样本，并与特征集合、PULP、CLAP 比较。当前仓库的旧切片原型仍可通过 `--method legacy-slice` 重现，但**不参与上述动态方法**。

```powershell
python -m fpspl.evaluation `
  --ranking E:\Myself\投稿\work7\FP-SPL\outputs\dynamic-bankaccount\ranking.csv `
  --labels-root E:\Myself\投稿\work7\pulp_trial\dataset
python -m fpspl.diagnostics `
  --graphs-root E:\Myself\投稿\work7\FP-SPL\outputs\dynamic-bankaccount `
  --labels-root E:\Myself\投稿\work7\pulp_trial\dataset
```

### 已完成的试跑

- `_MultipleBugs_.NOB_1.ID_98`：34 个配置、37 条失败测试、207 条通过测试；12 个原本通过的配置全部获得排序，重放／构图错误为 0。该版本没有可用的真假通过研究标签。
- `_MultipleBugs_.NOB_1.ID_194`：34 个配置、17 条失败测试、386 条通过测试；22 个原本通过的配置全部获得排序，重放／构图错误为 0。使用**排序后才读取**的研究标签评估，4 个假通过排在前 4 位，平均精度为 1.0。但只比较特征集合或条件分支也能达到 1.0，因此这一次试跑**没有证明动态图提高了识别能力**。

## 适用边界

- BRFL 的因子边是当前单条测试执行时的定义／使用和条件前驱关系。重复执行会在用于学习的行级图中合并，原始事件顺序保留在 `graph.tsv`。
- 如果测试重编译或插桩改变了测试结果，该测试被记录为错误并排除，不会用覆盖文件的预期结果冒充实际运行结果。
- 当故障位于所有配置共有的基础特征，且缺少能区分真实行为的测试或状态干预时，配置之间仍可能不可识别。此方法保留基础特征的证据，但不能保证识别所有这类故障。
- 输出目录可能很大，默认被版本控制忽略。源代码与实验配置应在独立分支提交。
- 当前评估只覆盖原本通过的配置；由失败配置删去失败测试而转换的假通过样本，需要单独构造并重放其保留测试，不能直接把原失败轨迹当成通过证据。

## 移植来源

`vendor/brfl/src/main/java/ppfl` 的 130 个 Java 源文件及代理清单取自本机 `E:\Myself\投稿\work5\code\BRFL`；`vendor/brfl/lib` 保存当时该仓库运行所需的依赖包。`fpspl/recompile.py` 从 BRFL 的 `recompile_failed_products.py` 移植了调试信息重编译流程，并改为在本项目输出目录中分别编译产品和测试，使通过配置也可采图。FP-SPL 的 `FpsplTraceExport.java` 将逐测试运行因子导出为本项目的图文件。重新编译、代理插桩和因子图构建均从本仓库启动，不写回原 BRFL 目录。
