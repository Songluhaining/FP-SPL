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
| 9 | `cfp_spl/causal.py` | 后验差异、语句筛选、证据等级和分数 |
| 10 | `cfp_spl/domain.py` | 上述模块共享的数据结构 |

## 主流程

1. 从配置报告读取特征选择和原始测试结果，仅将通过配置A作为待识别对象。
2. 为A选择最近的失败配置B，距离相同时按配置名确定顺序。默认只取一个B。
3. 对B的每个失败测试寻找A的对应测试。先确定性匹配，缺失时请求模型选已有测试，再缺失时生成测试。
4. 生成配对计划后重编译、执行。生成测试会额外编译进A和B；必须先在B上复现失败。
5. 生成测试在A也失败时直接记为确认假通过。A通过时，分别获得A/B的因子图及后验。
6. 将B的高可疑语句与A对齐，检查源码存在、实际执行、结果路径可达及后验差异。
7. 用固定规则为配对分级；配置取最高证据等级下的最佳配对分数进行排序。

## BRFL代码位置

所有移植代码在`vendor/brfl/`，不会调用或修改原BRFL仓库。建议沿以下位置检查：

- `vendor/brfl/src/main/java/ppfl/instrumentation/`：字节码插桩与日志记录。
- `vendor/brfl/src/main/java/ppfl/ByteCodeGraph.java`：日志解析、依赖及因子构建、`bp_inference()`推理。
- 同目录的`StmtNode.java`、`Node.java`、`FactorNode.java`：语句节点、运行值节点和因子。
- `java/ppfl/CounterfactualInferenceExport.java`：本项目增加的桥接导出程序。

## 必须注意的当前实现边界

当前代码是恢复版加大模型测试补全。它分别对A/B构图推理，不是联合因子图；尚未接入自适应区域张量消息传递。运行目录的`infer.txt`固定写入10个0.5参数，未接入频谱先验。配置排序也没有训练弱监督模型，使用的是`causal.py`中的固定阈值和权重。

确定性匹配接受的“相似测试”直接进入执行，不再让模型复核。模型只处理缺失配对。对于非生成测试，B执行原失败测试、A执行匹配测试；程序并没有把A所选测试移植到B执行。对于生成测试，两侧执行同一份生成源码。生成测试的B侧图导出异常时，会回退到B的原失败测试图，此时两侧测试不再相同。

`--plan-only`只跳过Java执行，不保证不访问大模型。接口密钥可以自动发现，因此人工阅读阶段不应把它当成离线选项。

模型响应缓存目前按输出目录及测试名寻找已有响应，并非完整请求内容摘要校验。若修改源码、模型或提示后复用同一输出目录，应人工检查缓存是否仍适用。结果执行缓存也有类似的复用限制。

图中依赖路径只提供动态依赖证据，并不等价于真实干预验证。生成测试导致A失败表示新测试发现故障现象，仍需人工审查测试判定条件是否有效，以及与评价数据的假通过定义是否一致。

## 输出和辅助入口

每个缺陷版本下的`pair-plan.json`用于检查近邻、测试来源和配对理由；`pair-assessments.json`包含逐语句证据；`errors.json`保存运行异常；`ranking.csv`保存配置排序。无有效配对的配置仍列在排序中，等级为证据不足，不代表已证明真通过。

`cfp_spl/batch.py`是批量分片入口；`evaluation.py`、`diagnostics.py`、`ablation.py`是结果分析入口；`experiments/evaluate_email_baselines.py`用于运行Email的PULP/CLAP基准。这些文件与主识别入口分开阅读即可。主识别过程不读取真假通过评价标签。
