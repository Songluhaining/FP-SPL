# FP-SPL：基于逐测试动态依赖图的假通过配置排序

当前主流程直接使用 BRFL 的字节码插桩和运行因子图，不读取 PULP 的切片日志。输入中的 `__FAILED__` 只作为已知故障行为样本，`__PASSED__` 只作为无标签候选；真假通过研究标签不进入训练。

## 实现流程

1. 从本地 BRFL 源码编译插桩代理和图构建库；不修改 BRFL 仓库。
2. 将每个配置的产品源码与测试源码以 `javac -g` 重新编译到输出目录，保留字节码到源码的行号映射。原始数据集保持只读。
3. 对每条测试分别启动 Java 虚拟机，使用 BRFL 代理记录 `opcode` 轨迹与 `.source.log`；核对实际测试结果与覆盖报告。
4. 使用 BRFL 的 `ByteCodeGraph` 对单条测试的源码日志和运行日志构图。导出每个运行因子的定义、使用与条件前驱，形成有方向且标注 `DATA`／`CONTROL` 的直接依赖边；保留事件数量和运行值。
5. 按覆盖文件中的 `featureClass`、`featureLineNum` 把源码位置映射至特征来源。以测试断言为可观察点，对依赖图取后向闭包，得到失败测试的局部行为模板和通过测试的局部图。基础特征也正常保留。
6. 将执行节点、带类型的依赖边、条件结果、源码词项和数值类别编码为图特征。用失败测试作正样本、通过测试作可能混有假通过的无标签背景，训练 8 个重复抽样的加权逻辑回归模型。候选测试优先由未把它抽入无标签背景的模型评分，再结合最相近失败模板的图相似度，按配置内最高测试分数排序。

此分数是**疑似假通过排序分数**，并非校准概率或最终真假判定。模型目前是可复现的弱监督图特征模型，尚未包含代码语义大模型或因果干预。

## 运行条件

- Python 3.10 及以上，安装 `numpy`、`scikit-learn`。
- Java 8 开发工具包。使用 BRFL 现有依赖包目录和本地源码。
- BankAccountTP 原始数据集有 `src/`、`test/`、逐测试覆盖及 `config.report.csv.done`。

```powershell
cd E:\Myself\投稿\work7\FP-SPL
python -m fpspl --method dynamic `
  --dataset D:\dataset\whn\4wise-BankAccountTP-1BUG-Full `
  --brfl-root E:\Myself\投稿\work5\code\BRFL `
  --java-home 'E:\Program Files\Java\jdk-1.8' `
  --output E:\Myself\投稿\work7\FP-SPL\outputs\dynamic-bankaccount `
  --version _MultipleBugs_.NOB_1.ID_98
```

去掉 `--version` 会依次处理全部缺陷版本。初次检查可用 `--limit-failed-variants 2 --limit-passing-variants 2 --max-tests-per-variant 2`；这些限制会改变训练样本和排序范围，不能当作完整实验结果。

## 输出与核查

- `<版本>/instrumented/<配置>/build/`：在工作区重编译的字节码。
- `<版本>/instrumented/<配置>/tests/<测试>/`：单条测试的代理轨迹、构图结果 `graph.tsv` 和实际结果 `status.json`。
- `<版本>/test_graphs.jsonl.gz`：逐测试的局部动态图；`typed_links` 是带方向和类型的直接运行依赖边。
- `<版本>/failure_templates.json`：失败测试的局部故障行为模板。
- `<版本>/pu_model.pkl`：已训练模型，只应从可信的本地输出加载。
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
