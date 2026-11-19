# 软件产品线假通过识别（FP-SPL）

当前方法是逐案例独立的联合贝叶斯推断。仓库只保留当前识别、CLAP/PULP比较、配置报告修正后的定位接入及结果核验；旧BRFL工程和早期方案源码已从工作树删除，可通过历史标签 `archive/pre-refactor-20261119` 查看。

## 运行

在仓库根目录使用本机Python解释器：

```powershell
python -m fp_research detect --system all --stratum all --list
python -m fp_research detect --system bank --stratum 2 --version _MultipleBugs_.NOB_2.ID_30
```

第一条只查看清单，第二条运行一个案例。默认每次建立独立人工结果目录，原始数据不变。数据路径见 `configs/datasets.json`；外部工具路径见 `configs/tools.json`。主方法最小依赖见 `requirements.txt`，比较及定位依赖见 `requirements-experiments.txt`。

- [代码架构](docs/ARCHITECTURE.md)
- [六系统人工运行、比较及定位命令](docs/RUNNING.md)
- [重构范围与核验记录](docs/REFACTOR.md)
- [历史语句级实验结果入口](experimental_results/语句级实验结果入口.md)

## 目录

活动源码统一在 `fp_research/data`、`detection`、`baselines`、`localization`、`experiments`、`evaluation`；测试统一在 `tests`，Java切片辅助程序在 `tools/java`。`fp_research/vendor/varcop` 是固定官方版本的只读源码快照。`experimental_results` 保留研究结果，`result` 保留既有工具与历史中间数据。

当前六系统为银行账户、邮件、电梯、考试数据库、图处理和压缩；前四个支持单、双、三缺陷，后两个多缺陷仍暂缓。每个案例独立初始化，不跨案例训练主方法；比较方法的有监督与正例学习流程单独保存版本划分。

```powershell
python -m unittest discover -s tests -v
python -m fp_research archive-check
```
