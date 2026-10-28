# BankAccountTP 全部 73 个缺陷版本的对照实验

本目录保存实验脚本，不把数据集、逐测试轨迹和训练权重纳入版本控制。原始数据集与原 PULP 仓库保持只读；PULP 的标注、切片和特征文件写在 `work7/pulp_trial/dataset` 副本中。

## 1. PULP：纳入基础特征缺陷

`pulp_all73.py` 先调用原 PULP 的标注和 50 项特征提取，再由 `pulp_include_base.py` 补充原代码跳过的基础特征缺陷版本。补充标注沿用原 PULP 的规则：对基础特征，所有原本通过配置都进入候选集；执行了故障语句的配置标为假通过。与原代码唯一有意不同的是，即使一个版本没有真通过配置，也保留其标签文件。30 个基分类器及其加权概率沿用原 PULP 实现；假通过／真通过标签仅在标注与评估中使用，不输入其弱监督分类器。

```powershell
cd E:\Myself\投稿\work7\FP-SPL
python experiments\pulp_all73.py
```

输出在 `work7/pulp_trial` 下：`base_case_labeling.json`、`predictions_system.csv` 和 `metrics_system.json`。

## 2. 本方法：全部版本的动态图

不限制配置数或逐测试数，12 个进程并行处理；每个缺陷版本单独训练图模型。中断后重新执行同一命令会复用已完成版本和逐测试轨迹。

```powershell
D:\SoftInstall\envs\orcalab_lerobot\python.exe -m fpspl.run_all `
  --dataset D:\dataset\whn\4wise-BankAccountTP-1BUG-Full `
  --java-home 'E:\Program Files\Java\jdk-1.8' `
  --output outputs\all73-graph --jobs 12 --model graph
```

## 3. 同口径比较

本方法只对原本通过的配置排序。对照时按“缺陷版本＋配置”匹配 PULP 的同一批原本通过配置，报告每版本排序平均精度和前 5 名精确率。PULP 的二值判定另报识别精确率、召回率；本方法尚无预先确定的判定阈值，不能把其排序分数直接当二值判定。

```powershell
python experiments\compare_all73.py `
  --dataset ..\pulp_trial\dataset `
  --pulp ..\pulp_trial\predictions_system.csv `
  --graph outputs\all73-graph\ranking.csv `
  --base-labels ..\pulp_trial\base_case_labeling.json `
  --output outputs\all73-comparison
```

基础特征缺陷组的 19 个版本在 PULP 标注下均无真通过配置。因此该组的排序平均精度恒为 1，不能作为区分能力证据；应单列其覆盖率与二值召回率。
