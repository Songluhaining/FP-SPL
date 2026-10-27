# FP-SPL：软件产品线假通过配置的局部图排序原型

本仓库建立一个可复现的第一阶段：从**现有测试未通过的配置**提取逐测试故障行为模板，再将**现有测试通过的配置**按其执行证据与模板的关系排序。排序只使用测试结果和运行证据，不读取真通过／假通过的实验标签。

## 当前输入及证据含义

- `config.report.csv.done` 中的 `__TEST_OUTPUT__` 只用于分出已知失败配置和待排序的通过配置。
- `coverage/failed/*.coverage.xml` 和 `coverage/passed/*.coverage.xml` 是逐测试执行覆盖；条件行中的 `truecount`、`falsecount` 提供已走过的分支。
- `spectrum_*_coverage.xml` 中的 `featureClass`、`featureLineNum` 把生成程序中的行映射回原始特征，沿用 BRFL 的特征位置映射思路。基础特征也作为普通特征保留。
- `slice.all_statements.backward_slicing.log` 提供语句的**传递性后向切片集合**。只有两个语句都在某条测试中执行过时才保留其可达关系。它不是 BRFL 的逐事件直接动态依赖边，也没有执行顺序和运行值；结果文件明确标为 `coverage+slice-reachability`。
- 测试源码中的断言记录预期表达式和测试可观察点；覆盖数据并不包含失败时的实际运行值。因此模板中的 `observations` **不是实测输出值**。

本仓库不复制 BRFL 代码，也不改动其仓库。当前原始 BankAccountTP 数据没有 BRFL 逐测试轨迹，故这个可运行原型使用此前 PULP 试跑生成的语句切片。接入 BRFL 逐事件图后，应保留边的类型、方向、运行值与测试身份，再替换这里的切片可达关系。

## 运行

需要 Python 3.10 或更新版本；当前实现只使用标准库。以下命令在已有的 BankAccountTP 原始数据和 PULP 试跑切片上运行一个缺陷版本：

```powershell
cd E:\Myself\投稿\work7\FP-SPL
python -m fpspl `
  --dataset D:\dataset\whn\4wise-BankAccountTP-1BUG-Full `
  --slice-dataset E:\Myself\投稿\work7\pulp_trial\dataset `
  --output E:\Myself\投稿\work7\FP-SPL\outputs\bankaccount `
  --version _MultipleBugs_.NOB_1.ID_98
```

移除 `--version` 即运行数据集中的全部缺陷版本。若缺少切片，程序默认报错；只有显式传入 `--allow-coverage-only` 才会退化为仅覆盖排序，且其结果没有依赖关系证据。

## 输出

- `ranking.csv`：每个缺陷版本内，对测试通过的配置排序。`score` 是图相似和失败样本富集形成的**排序分数**，不是已校准的假通过概率，也不是最终真假标签。
- `<版本>/failure_templates.json`：每条可用失败测试的语句、切片可达关系、执行分支、特征来源、源码和断言。
- `<版本>/test_graphs.jsonl.gz`：逐测试局部执行图，包括用于构建模板的失败测试以及用于排序的通过测试。
- `<版本>/summary.json`：样本数、图证据等级和缺失数据。

训练权重来自失败测试图与无标签通过测试图的**出现频率对比**。通过测试只是背景分布，未被赋予“真通过”负标签。每个通过配置取其测试图与故障模板之间的最佳匹配；缺少图或缺少模板时，标记为 `insufficient_evidence`。此阶段是透明的弱监督排序器，**尚不是深度模型或完整的因果识别方法**。

若有单独的研究标签，可在**排序完成后**运行评估程序。它只对原本就通过的配置计算排序指标，不将标签送回排序器：

```powershell
python -m fpspl.evaluation `
  --ranking E:\Myself\投稿\work7\FP-SPL\outputs\bankaccount\ranking.csv `
  --labels-root E:\Myself\投稿\work7\pulp_trial\dataset `
  --output E:\Myself\投稿\work7\FP-SPL\outputs\bankaccount\evaluation.json
```

可用 `python -m fpspl.diagnostics --graphs-root <输出目录> --labels-root <标签数据目录>` 检查“只比特征集合”这一简单方法是否已经足够。诊断程序与正式排序程序分离。

## 实验边界

1. 当前图可以提供结构相似性证据，不能单凭切片证明因果关系或判断代码行为正确性。
2. 当基础特征中的缺陷存在于所有配置，或缺陷路径没有被任何通过测试执行，配置对比可能缺乏可识别信号；需要后续的测试输入或状态干预。
3. 原始失败配置转成假通过配置时，必须隐藏被移除的失败测试及其轨迹。程序仅对报告中通过的配置读取 `coverage/passed`，不会把其 `coverage/failed` 纳入排序输入。
4. 如果使用实验标签评估，应按缺陷版本划分数据，并分开报告原本通过和转换而来的假通过配置；不要用同一批样本的真假标签调整排序权重或阈值。

### BankAccountTP 初步诊断（2026-10-27）

73 个缺陷版本中，72 个产生了模板；1666 个原本通过的配置获得排序，1 个版本因模板不足没有排序。在有研究标签且同时含真假通过配置的 38 个版本上，本排序器的平均排序精度约为 **0.973**；但**仅比较特征集合**的简单方法达到 **1.000**。这说明当前数据及标签划分并不能证明依赖关系带来了增益，甚至这一初版组合分数略低于简单特征匹配。当前结果只用来检查管线和暴露评估问题，不能作为优于 PULP 或 CLAP 的论文结论。基础特征缺陷版本缺少相应研究标签，也未纳入该数字。

## 版本控制

远程仓库为 `https://github.com/Songluhaining/FP-SPL.git`。`outputs/` 和 `data/` 被忽略，以免提交大体量数据及生成结果。后续大规模改动应在独立分支实施、运行验证并提交，再合并到主分支。
