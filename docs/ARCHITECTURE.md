# 代码架构

当前仓库只维护联合贝叶斯假通过识别、CLAP/PULP比较、修正配置报告后的定位对照及结果核验。旧BRFL因子图工程、早期单故障推断、六步深度学习方案、补测与历史控制脚本已经从当前工作树删除，历史源码保留在 `archive/pre-refactor-20261119` 标签中。

```text
BRFL/
├─ configs/                  数据路径、外部工具路径
├─ fp_research/
│  ├─ __main__.py            唯一推荐命令入口
│  ├─ config.py              路径解析
│  ├─ io.py                  原子文件、摘要、运行名校验
│  ├─ data/                  原始报告、覆盖读取及六系统案例选择
│  ├─ detection/             当前联合贝叶斯方法
│  ├─ baselines/             CLAP/PULP特征与分类适配
│  ├─ localization/          固定版VARCOP、特征组合、切片和报告修正
│  ├─ experiments/           定位实验计划、预测来源、运行和汇总
│  ├─ evaluation/            评价真值、分折、排名、独立统计及归档检查
│  ├─ vendor/varcop/          未改动的官方源码快照
│  └─ manual_joint.py        上一版人工命令的轻量兼容入口
├─ tests/                    当前功能的测试
├─ tools/java/               两个实际使用的静态切片辅助程序
├─ docs/                     架构、运行和重构核验说明
├─ experimental_results/     已归档结果和新实验结果
└─ result/                   已有工具下载及历史中间数据，不是活动源码
```

## 主方法

|文件|作用|
|---|---|
|data/catalog.py|按六系统、缺陷数和版本选择案例，检查报告基本格式|
|data/inventory.py|核对配置、源码、测试源码和逐测试成败目录|
|data/coverage.py|读取汇总覆盖，核对源码身份和原始配置状态|
|detection/evidence.py|读取逐测试覆盖；合并覆盖列完全相同的候选语句；构建配置成员关系|
|detection/model.py|设置先验，执行序贯蒙特卡洛采样，计算联合后验和配置风险|
|detection/runner.py|逐案例初始化，核对两种种子并升级粒子数，保存结果和摘要|
|detection/cli.py|人工选择案例，隔离本次输出并生成修正报告|

主方法调用关系为：

```text
detect命令
  → data.catalog选择案例
  → detection.runner.fit
      → detection.evidence.load_evidence
          → data.inventory / data.coverage
      → detection.model.infer_joint
          → prior_table / conditional_likelihood / risks
  → 保存逐配置结果、粒子状态和独立修正报告
```

`detection/model.py` 与重构前模型逐字节一致。每个案例重新初始化，不使用其他案例的后验或真假通过标签。目录中的缺陷数仅用于选例和记录，不作为已知故障数量传入模型。当前固定设置仍为种子42、2026；9个非零触发率加0；稀疏参数1.5；粒子数256→1024→4096；稳定性差异阈值0.05；分类阈值0.5。

## 比较与定位

比较方法的静态切片、特征适配、已核验缓存公式及公开分类器调用分别位于 `baselines/slicing.py`、`features.py`、`cache.py`、`classifier.py`。比较方法按系统及缺陷层固定版本五折，监督方法只把训练折真值交给分类器；正例学习保留等量抽样和默认5%两种设置。分类至少需要同系统同缺陷层两个版本，不能用单案例训练自身。

定位层只保留配置报告修正方式：新增每种方法的报告，只把预测假通过配置最后一列改为 `__FAILED__`，保留全部配置和原始覆盖。特征组合和基础特征判断读取新报告；产品级覆盖评分仍遵循原版实际覆盖文件，不伪造失败测试。

`experiments/catalog.py` 仅解析预测来源，不训练模型。定位默认明确引用此前已归档的预测；提供 `--detection-run` 和 `--predictions-dir` 才使用指定的新识别和基线输出。当前输出保存到独立的新实验目录，不能覆盖历史主表。

## 历史可追溯性

`experimental_results` 中重构前已有文件保持原字节。其源码摘要描述对应历史提交，不能拿重构后的文件路径直接替代旧源码。需要逐字节复核旧源码时，从历史标签或结果版本清单指定的Git提交读取。

归档核验命令先检查实际结果文件；历史源码已移动或删除时，才从对应提交读取源码摘要。结果文件摘要不符会报错，不会以Git中的旧结果掩盖工作树结果被修改。
