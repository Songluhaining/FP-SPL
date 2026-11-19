# 当前假通过识别方法的人工运行说明

## 仓库与当前入口

本地仓库为 `E:/Myself/投稿/work7/BRFL`，远程为 https://github.com/Songluhaining/FP-SPL.git ，研究分支为 `research/fp-detection`。本地文件夹沿用旧名称，但当前主方法是逐案例独立的联合贝叶斯推断。

人工识别入口为 `python -m fp_research.manual_joint`。它调用已用于完整实验的 `multifault_study.fit()`、`joint_inputs.load_evidence()`、`joint_fault.infer_joint()`，不修改推断算法或参数，也不调用早期六步方案、原BRFL因子图入口、比较方法训练或语句级定位。

## 六系统及数据路径

|系统|命令中的系统代码|数据根目录|当前纳入的缺陷层|
|---|---|---|---|
|银行账户|bank|D:/dataset/whn/BankAccountTP|1、2、3|
|邮件|email|F:/datasets/Email-FH-JML|1、2、3|
|电梯|elevator|F:/datasets/Elevator-FH-JML|1、2、3|
|考试数据库|examdb|F:/datasets/ExamDB|1、2、3|
|图处理|gpl|D:/dataset/whn/GPL|1|
|压缩|zipme|D:/dataset/whn/ZipMe|1|

路径统一读取 `fp_research/dataset_locations.json`。双、三缺陷层的图处理与压缩系统仍暂缓，选择全部案例时也会跳过它们。当前清单共1054个案例：银行账户383、邮件126、电梯122、考试数据库263、图处理105、压缩55。

标准案例路径为 `数据根目录/4wise-系统目录名-缺陷数BUG-Full/_MultipleBugs_.NOB_缺陷数.ID_编号`。命令中的缺陷数仅用于选择目录及记录结果，不传给推断函数作为已知故障数量。

## 本机准备

在PowerShell中执行一次：

```powershell
Set-Location 'E:\Myself\投稿\work7\BRFL'
$researchPython = 'D:\SoftInstall\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONHASHSEED = '42'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $researchPython -m fp_research.manual_joint --help
```

本机已核实的运行环境为Python 3.11.4、数组计算库NumPy 1.24.3、科学计算库SciPy 1.10.1、数值线程控制库threadpoolctl 2.2.0。此识别入口使用已有覆盖，不重新编译Java、不重新执行测试，也不需要深度学习模型。

## 先查看案例，再选择运行

下列命令只列出清单，不运行推断、不生成案例文件：

```powershell
# 查看六系统全部单缺陷案例。
& $researchPython -m fp_research.manual_joint --system all --stratum 1 --list

# 查看某个系统的双缺陷案例，替换系统代码即可。
& $researchPython -m fp_research.manual_joint --system bank --stratum 2 --list

# 查看当前协议下六系统所有可运行案例，应该为1054个。
& $researchPython -m fp_research.manual_joint --system all --stratum all --list
```

选择一个真实案例进行推断，例如银行账户双缺陷案例30：

```powershell
& $researchPython -m fp_research.manual_joint --system bank --stratum 2 --version '_MultipleBugs_.NOB_2.ID_30'
```

也可以直接指定该案例的完整路径：

```powershell
& $researchPython -m fp_research.manual_joint --case 'D:\dataset\whn\BankAccountTP\4wise-BankAccountTP-2BUG-Full\_MultipleBugs_.NOB_2.ID_30'
```

运行某个系统、某个缺陷层的所有案例时，省略 `--version`：

```powershell
& $researchPython -m fp_research.manual_joint --system email --stratum 3
```

确实需要由你启动全量识别时，使用：

```powershell
& $researchPython -m fp_research.manual_joint --system all --stratum all
```

程序串行处理所选案例，每个案例独立读取证据、初始化粒子并推断，不使用其他案例的后验或真假通过标签。此命令不训练CLAP或PULP，不运行VARCOP。

## 本次输出和缓存规则

不指定运行名时，每次命令自动生成新的时间及随机标识，真实重新计算所选案例，保存到：

```text
案例目录/
  .fp_research/
    manual_joint_detection/
      本次运行名/
        inventory.json
        evidence.json
        evidence.npz
        posterior-states.npz
        joint.csv
        joint.json
        config.report.joint.csv
        manual-run.json
```

终端会打印本次完整结果路径。`joint.csv` 是逐原本通过配置的风险和预测：`score` 为风险，`prediction=1` 表示达到固定0.5阈值；`stable` 表示两种种子是否满足既定稳定性条件。`lower`、`upper` 是观测等价语句在不同位置分配下的风险界，不是采样误差置信区间。

`joint.json` 保存参数、粒子数升级记录、两次采样差异、采样过程及产物摘要。`posterior-states.npz` 保存两个种子的最终粒子状态。`config.report.joint.csv` 是本次识别对应的新报告，只把预测假通过配置最后一列由 `__PASSED__` 改成 `__FAILED__`。原始报告和根目录中前轮导出的同名方法报告都不覆盖。

如需复用你这次人工运行的缓存，显式指定运行名，例如：

```powershell
& $researchPython -m fp_research.manual_joint --system bank --stratum 2 --version '_MultipleBugs_.NOB_2.ID_30' --run-name check-bank2-30
```

第一次会计算；再次使用相同运行名、相同案例且源码与输入摘要一致时，才会显示 `cached`。要重新计算，省略运行名或换一个新名字，不需要删除任何历史结果。

退出码0表示所选案例均成功且稳定，1表示有运行失败，2表示无运行失败但有案例未满足稳定性条件。未稳定结果仍保留标记，不当作稳定结果发布。

## 当前方法的代码架构

|层次|文件及主要函数|职责|
|---|---|---|
|人工入口|manual_joint.py：select_jobs、run_job|选择六系统案例，隔离本次输出，调用现有推断，生成对应修正报告|
|逐案例执行|multifault_study.py：fit|保存输入；分别使用种子42、2026推断；必要时增加粒子；保存逐配置结果和核验信息|
|输入清点|inventory.py：records|核对配置、启用特征、源码、测试源码及逐测试成败目录是否一致|
|源码身份与汇总覆盖|latent_fault.py：read_version|把覆盖映射到特征源码语句，并核对原始报告与覆盖；当前调用链只使用读取函数，不调用该文件的早期单故障infer|
|证据构建|joint_inputs.py：load_evidence|构建逐测试语句覆盖矩阵；把覆盖列完全相同的语句合并；形成配置与语句组的成员关系|
|联合推断|joint_fault.py：prior_table、infer_joint|给定稀疏先验与触发率取值，依次加权、重采样、更新粒子，保留多个故障并存的解释|
|配置风险|joint_fault.py：risks|对每个粒子中的实际故障组合计算配置风险，再在粒子之间求平均，输出风险及界|
|比较方法|baseline_features.py、compare.py|另外生成CLAP、PULP特征并调用其分类实现；不参与主方法的逐案例后验推断|
|定位接入|varcop_report_relabel.py、report_relabel_study.py|读取识别结果生成独立配置报告并进行定位对照；不是假通过识别核心|
|核验|checks/test_joint_fault.py、checks/test_manual_joint.py|分别核验联合推断的小图结果与人工入口的选例、输出和报告修改边界|

主要调用关系为：

```text
manual_joint.run_job
  -> multifault_study.fit
       -> joint_inputs.load_evidence
            -> inventory.records
            -> latent_fault.read_version
       -> joint_fault.infer_joint
            -> prior_table / conditional_likelihood / risks
       -> 保存joint.csv、joint.json和粒子状态
  -> manual_joint.corrected_report
       -> 保存本次config.report.joint.csv
```

## 当前固定设置与输入边界

当前设置为9个非零触发率加0，共10种状态；稀疏先验参数1.5；两种种子42和2026；从256个粒子、两轮更新开始。两种种子对全部候选的最大分数差超过0.05时，依次升级到1024个粒子、三轮更新和4096个粒子、四轮更新。分类阈值固定为0.5。

识别始终读取未修正的 `config.report.csv.done` 和原始逐测试覆盖。不要把生成的修正报告替换成识别的原始输入，否则会把预测重新作为观察证据。识别不读取变异日志、真实故障位置或真假通过评价标签。每个案例重新初始化；单、双、三缺陷使用相同推断函数。

如果后续人工运行VARCOP，请明确选择本次输出目录中的修正报告，并重新生成对应的特征组合与切片。案例根目录现有的 `config.report.joint.csv`、`config.report.CLAP.csv` 等是之前完整实验导出的报告，不会随着本次人工识别自动更新。

## 与历史入口的区别

`python -m fp_research infer`、study.py、fit.py、risk.py及README下方保留的六步流程属于早期方案。joint_pilot.py属于先导实验，latent_fault.infer属于早期单故障模型；这些都不是本次推荐的人工运行入口。localization_study.fit是完整定位实验的准备阶段，会按协议复用历史推断，不适合据此判断是否真实重新采样。

本次新增的是人工运行外层，核心joint_fault.py、joint_inputs.py、multifault_study.fit及历史实验结果均保持不变。本次交付只核验入口和清单，没有替用户重新运行真实案例推断。
