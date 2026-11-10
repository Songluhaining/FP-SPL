# 软件产品线假通过识别实验

本目录的入口在原有因子图工具内增加配置识别实验。主实验只预测原本通过的配置。缺陷版本是训练与测试的划分单位，每个系统做五折验证，另做两次跨系统迁移。原缺陷定位命令仍由仓库根目录的脚本（main.py）提供。

## 在本机复现

在仓库根目录依次运行以下两条命令。主实验提供普通失败栈与已核验因子图，补充实验可复用输入指纹一致的图。两次运行的默认输出分别位于实验结果目录（experimental_results/fp_detection/main、masked）。

```powershell
python -m fp_research.study --scenario main --model-python "../FP-SPL/.venv/Scripts/python.exe"
python -m fp_research.study --scenario masked --model-python "../FP-SPL/.venv/Scripts/python.exe"
```

默认原始数据在 D 盘数据目录（D:/dataset/whn）。基准源码目录（CLAP、PULP）、已有静态切片目录（pulp_trial/dataset、email_trial/dataset）和当前仓库位于同一上级目录。切片按配置与原始产品源码核对，基准特征与分类结果重新计算。补充样本从已有转换清单取得配置标识，移除失败用例与派生覆盖，再重跑剩余测试。

本次构建采用第八版开发工具（Java 1.8.0_381）。特征与分类使用解释器 3.11.4 版（Python）、数组运算库 1.24.3 版（NumPy）、表格计算库 1.5.3 版（pandas）、机器学习库 1.3.0 版（scikit-learn）和语法解析库 0.13.0 版（javalang）。集合网络使用相同版本的解释器、深度学习库 2.6.0 版（PyTorch）及数组运算库 2.4.6 版。其训练在中央处理器上进行。两套解释器分别由命令行的当前解释器和模型解释器参数（--model-python）选择。

可通过阶段参数（--start、--end）指定连续步骤，依次为准备数据、推理、训练模型、计算基准特征、基准分类、配置排名、实际补测和独立复算（prepare、graph、model、features、classify、rank、replay、audit）。查询单步用法或只跑一个版本可使用以下命令。

```powershell
python -m fp_research --help
python -m fp_research infer --help
python -m fp_research inventory --system bank --version _MultipleBugs_.NOB_1.ID_1 --data D:/dataset/whn --output result/quick-check
```

## 判定与核验

- 测试语法树提供操作、输入和断言。同一失效组还要求覆盖相近。普通执行的异常栈映射到特征源码行或测试断言，组内出现多个已知失效位置时拆组，缺少位置时保留缺失状态。
- 同配置通过用例须有共同的观察目标和一致的被测操作集合。同测试类优先，输入相似度与稳定的用例标识用于处理并列。缺少对照的失败证据单独保留。
- 每个组用原生插桩重放失败和对照用例。指定单测时仅对相应测试方法插桩，保留原方法、源码行号及脚手架。用例调用其他测试方法时保留整个类。异常替身复核仅接受已从依赖核实的非法参数异常对应关系，并要求相同产品或断言位置。每次执行核对实际方法、成败与可取得的异常位置，轨迹不可用时独立重试一次，跨配置仅共享可映射的特征语句。正确后验的补数作为可疑度，按目标构建与启用特征汇总权重。汇总表中的零执行次数与未包含该语句分别处理。
- 学习检出能力时，在已知失败配置中抽取不同大小的测试子集。参照失效描述来自该组的其他配置。模型编码测试与参照在操作、输入、断言上有多少重合，用注意力汇总测试集合，并用实际子集成败训练。单配置失效组缺少独立参照，不生成训练子集。
- 主模型按组计算适用分数乘以预测漏检程度，取配置内最高组分。分类阈值固定为 0.5，同时报告摆脱固定阈值的平均精确率。某些组构图失败时保留部分证据标记。完全不能打分时保留配置，二分类表按未报出计。
- 三种补测选择策略在同一版本与同一配置的共同候选池中比较。候选来自可见的失败用例，先检查被测类型和操作是否可移植，再比较模型预测增益、随机顺序和新增可覆盖语句数。参考产品按相同特征从原始产品线模板构建。用例在参考产品通过，且目标产品在两个独立运行进程出现同一断言错误时记为确认。
- 主实验的对照标签按基准的语句出现规则重建，另报需要正执行次数的口径。监督基准只在训练折使用真假标签。正例与未标注基准使用原始失败配置为正例，代理负例从训练折的未标注配置有放回抽取，数量取该折的正例数。默认三十个分类器，随机种子为 42。
- 历史路径处理仅在加载监督基准的进程内统一路径分隔符。正例与未标注实现使用十种已有可疑度函数，并把其排名返回值转换为特征提取函数所需的语句到分数的映射。

完整审计从逐配置文件重新生成表格，并检查普通通过候选集合、版本划分、组内执行成员、参考构建结果、两次断言复现及隐藏用例隔离。版本清单中的摘要用于核对切片和输入来源。

## 静态切片与阶段检查

版本清单的源码摘要和切片摘要分别覆盖原始构建和两个方向的全语句切片。若需重新构造切片，把同一缺陷版本的配置目录（variants）复制到单独的实验目录，在该副本所在驱动器上执行基准附带的全语句切片程序。以下示例在切片工作副本的版本目录中运行，两个变量分别指向开发工具安装路径与基准源码路径。

```powershell
$variants = (Resolve-Path ".\variants").Path
& "$javaHome\bin\java.exe" -Xmx512m "-Dvariants_dir=$variants" -jar "$pulpRoot\plugins\all-statements-slicing.jar"
```

每个配置应产生前向与后向两份切片。两个方法各自重新计算版本特征，按固定版本划分拟合，再对留出配置评分。仓库中的分数对应本次两个系统的统一协议。

语义归组、对照配对、失效位置、构建匹配、风险和单测隔离可执行下列检查。网络置换检查使用装有深度学习库的解释器。

```powershell
python -m unittest fp_research.checks.test_intents fp_research.checks.test_pairing fp_research.checks.test_sites fp_research.checks.test_applicability fp_research.checks.test_risk fp_research.checks.test_isolated
& "../FP-SPL/.venv/Scripts/python.exe" -m unittest fp_research.checks.test_network
```
