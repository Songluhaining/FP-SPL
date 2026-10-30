# 当前版本代码阅读指南

## 入口和阅读顺序

终端执行`python -m cfp_spl`，先进入`cfp_spl/__main__.py`，再调用`cfp_spl/cli.py`的`main()`。人工调试建议从`cli.main()`设置断点，随后进入`pipeline.CounterfactualPipeline._run_version()`。

| 顺序 | 文件 | 检查内容 |
|---|---|---|
| 1 | `cfp_spl/cli.py` | 参数与默认值，主流程实例化 |
| 2 | `cfp_spl/pipeline.py` | 每个缺陷版本的完整执行顺序、异常与输出 |
| 3 | `cfp_spl/configurations.py` | 原始通过/失败结果、特征向量、欧式距离和近邻 |
| 4 | `cfp_spl/java_tests.py` | 测试发现、失败测试提取、内容及断言相似度 |
| 5 | `cfp_spl/generator.py` | 大模型选择、生成提示、响应检查与缓存 |
| 6 | `cfp_spl/brfl_runtime.py` | 重编译、插桩命令、测试结果、取图和推理调用 |
| 7 | `java/ppfl/CounterfactualInferenceExport.java` | 静态日志和动态轨迹转为BRFL因子图，调用推理并导出 |
| 8 | `cfp_spl/factor_graph.py`、`feature_sources.py` | 图解析、结果依赖路径、跨配置语句身份对齐 |
| 9 | `cfp_spl/joint_causal.py` | 共同结果路径、统一后验与配置评分；`causal.py`只保留直接失败和旧版兼容 |
| 10 | `cfp_spl/domain.py` | 上述模块共享的数据结构 |

## 主流程

1. 从配置报告读取特征选择和原始测试结果，仅将通过配置A作为待识别对象。
2. 为A选择所有并列最近的失败配置B；配置名只决定输出顺序。指定多个近邻时，也保留截断边界上的全部并列项。
3. 对B的失败测试寻找A对应测试：规则完全一致匹配，其他配对由模型判断场景、功能和断言目标；没有候选时生成测试。
4. 生成测试进行结构检查、独立语义复核、真实编译执行和诊断反馈修复。B必须因断言失败复现故障；A的普通异常不构成确认假通过。
5. A通过时先独立解析各执行结构，不运行单侧概率推理；按特征源码语句共享静态节点，保留各次执行独立动态值节点和因子。
6. 同一A的有效执行去重后共同进入一个图，只进行一次循环信念传播。要求图中同时存在通过和失败观测，语句身份没有重复。
7. 统一图中，A/B共同执行且结果依赖可达的语句最高可疑度用于排序。不使用旧版后验差值。模型选择的测试若动态结果路径无共同特征语句，则尝试生成替代测试。

## BRFL代码位置

所有移植代码在`vendor/brfl/`，不会调用或修改原BRFL仓库。建议沿以下位置检查：

- `vendor/brfl/src/main/java/ppfl/instrumentation/`：字节码插桩与日志记录。
- `vendor/brfl/src/main/java/ppfl/ByteCodeGraph.java`：日志解析、依赖及因子构建、`bp_inference()`推理。
- 同目录的`StmtNode.java`、`Node.java`、`FactorNode.java`：语句节点、运行值节点和因子。
- `java/ppfl/CounterfactualInferenceExport.java`：本项目增加的桥接导出程序。

## 必须注意的当前实现边界

当前已经使用统一图，不训练弱监督模型；仍使用BRFL循环信念传播，未加入区域张量消息传递。`infer.txt`的10个0.5是BRFL运算因子参数，不能简单理解成10个语句先验；当前未加入论文中的异质先验及特征节点，复用的是语句/动态值依赖核心。

非生成测试由B执行原失败测试、A执行配对测试；生成测试使用同一份测试源码在两侧运行。不再回退到不相干测试的B图。允许近似反事实场景，前提是模型复核保持功能与故障判定目标，真实轨迹又包含共同的结果路径语句。

`--plan-only`会跳过Java验证，但仍可能调用模型。模型响应缓存由请求、模型、服务、协议版本区分；执行目录加入源码摘要，修复测试不会误复用旧结果。`--max-passing`仅用于过程验证，正式全量实验应设为0。

覆盖映射中`name="EmailSystem.Client"`是完整类名，不是带扩展名的文件名；优先利用`path="EmailSystem/Client.java"`构造运行类名。冲突映射会报错，未映射节点按执行标识隔离，禁止跨配置猜测合并。

语句节点按特征源码行唯一，同行字节码指令共享语句变量；值节点按每次执行产生的运行值独立，不是每行强制只产生一个值。图依赖和模型复核均不是因果干预证明；结果仍需对测试语义和故障位置进行人工检查。

## 过程日志

- `planning-events.jsonl`：逐配对开始时间、A/B配置和距离。
- `generation-requests/`：各轮模型请求、响应、语义复核及编译执行诊断。
- `events.jsonl`：配对验证、轨迹相似度、替代生成、联合推理开始/完成和异常。
- `joint/<配置>/executions.tsv`：进入统一图的去重执行清单。
- `joint/<配置>/source-map-*.tsv`：每次执行的运行源码到特征源码映射。
- `joint/<配置>/joint-factor-graph.tsv`：带统一后验的完整图，观测值额外记录真假结果。
- `joint/<配置>/joint-inference.log`及`trace/logs/`：推理原始日志，含迭代信息。

`experiments/replay_joint_cached.py`回放已有轨迹，校验静态语句唯一性、动态值不被合并、映射集合一致以及真假观测共存。它没有补全旧版缺失的测试，结果不能代替新模型的完整实验。

## 输出和辅助入口

每个缺陷版本下的`pair-plan.json`用于检查近邻、测试来源和配对理由；`pair-assessments.json`包含逐语句证据；`errors.json`保存运行异常；`ranking.csv`保存配置排序。无有效配对的配置仍列在排序中，等级为证据不足，不代表已证明真通过。

`cfp_spl/batch.py`是批量分片入口；`evaluation.py`、`diagnostics.py`、`ablation.py`是结果分析入口；`experiments/evaluate_email_baselines.py`用于运行Email的PULP/CLAP基准。这些文件与主识别入口分开阅读即可。主识别过程不读取真假通过评价标签。
