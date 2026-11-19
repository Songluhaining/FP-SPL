# 人工运行

在 `E:/Myself/投稿/work7/BRFL` 根目录执行。当前本机解释器为 `D:/SoftInstall/python.exe`。识别依赖数组计算库NumPy、科学计算库SciPy、线程控制库threadpoolctl；比较与定位另需pandas、scikit-learn、javalang、xlsxwriter，以及公开分类实现导入的绘图库matplotlib。`requirements.txt` 和 `requirements-experiments.txt` 保存已验证的本机版本，不再携带旧项目无关环境。

```powershell
Set-Location 'E:\Myself\投稿\work7\BRFL'
$researchPython = 'D:\SoftInstall\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONHASHSEED = '42'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $researchPython -m fp_research --help
```

## 数据与工具

数据路径统一修改 `configs/datasets.json`。银行账户、图处理、压缩仍在 `D:/dataset/whn`；邮件、电梯、考试数据库在 `F:/datasets`。系统代码依次为 `bank`、`gpl`、`zipme`、`email`、`elevator`、`examdb`。银行账户、邮件、电梯、考试数据库包含1、2、3缺陷；图处理和压缩的多缺陷仍暂缓，`--stratum all`不会启动它们。

外部工具路径统一修改 `configs/tools.json`。CLAP、PULP使用仓库上级目录中的各自原始源码；静态切片需要Java8及PULP附带程序。当前识别只读取已有覆盖，不需要Java或深度学习库。VARCOP官方源码快照保留在 `fp_research/vendor/varcop`，切片程序固定提交的摘要见其 `SOURCE.json`。

需要准备或核验固定版VARCOP切片程序时，运行 `python -m fp_research.localization.source`。此工具按固定提交获取程序、校验摘要，并保存到工具配置指定的位置；普通识别不会调用它。

## 识别

```powershell
# 只列出当前1054个可用案例，不执行推断。
& $researchPython -m fp_research detect --system all --stratum all --list

# 单个真实案例。
& $researchPython -m fp_research detect --system bank --stratum 2 --version '_MultipleBugs_.NOB_2.ID_30'

# 指定运行名，方便稍后接入定位。
& $researchPython -m fp_research detect --system bank --stratum 1 --run-name check-bank1

# 也可以直接指定案例完整路径。
& $researchPython -m fp_research detect --case 'D:\dataset\whn\BankAccountTP\4wise-BankAccountTP-2BUG-Full\_MultipleBugs_.NOB_2.ID_30'
```

省略 `--version` 就运行该系统、该缺陷层的所有案例。`--system all --stratum all` 是全量识别命令，只有用户实际执行才会启动。

默认每次生成新的运行名并重新计算，保存到 `案例/.fp_research/manual_joint_detection/运行名/`。重复显式运行名时，仅在源码、输入和产物摘要都一致时复用该人工目录的缓存。不会复用历史整批推断来冒充重新采样，也不覆盖案例根目录前轮导出的报告。

|输出文件|用途|
|---|---|
|joint.csv|逐原本通过配置的风险、0/1预测、风险界和稳定性标记|
|joint.json|参数、粒子升级、两种种子差异、采样过程及产物摘要|
|evidence.json / evidence.npz|配置、语句组和逐测试证据|
|posterior-states.npz|两个种子的最终粒子状态|
|config.report.joint.csv|本次结果对应的独立修正配置报告|
|manual-run.json|本次人工入口及输出来源|

原始输入始终为 `config.report.csv.done` 和逐测试覆盖；修正报告不能反过来作为识别原始输入。`lower`、`upper` 表示观测等价位置分配下的风险界，不是采样误差置信区间。退出码0为成功且稳定，1为运行失败，2为存在未达到稳定性条件的案例。

上一版命令 `python -m fp_research.manual_joint ...` 仍可使用，它只是转发到当前识别入口。

## 比较方法

```powershell
# 生成一个系统一个缺陷层的切片及两种方法特征。
& $researchPython -m fp_research baseline prepare --system bank --stratum 1 --run-name compare-bank1 --method all

# 在同一批特征上按固定版本五折运行CLAP、PULP等量及PULP默认抽样。
& $researchPython -m fp_research baseline classify --system bank --stratum 1 --run-name compare-bank1 --method all
```

特征保存在各案例 `.fp_research/baselines/运行名/`，折外预测保存在 `experimental_results/baselines/运行名/fold_predictions/`。不同方法各自的参数、源代码摘要、训练/测试版本及特征摘要保存在折文件旁的JSON中。准备特征不使用真假通过评价标签；CLAP训练阶段读取训练标签，PULP使用可见失败和未标注配置。

## 定位对照

```powershell
# 使用上述同名人工识别结果和新生成的折外预测。
& $researchPython -m fp_research localize --system bank --stratum 1 --run-name locate-bank1 --detection-run check-bank1 --predictions-dir 'experimental_results\baselines\compare-bank1\fold_predictions'

# 不提供这两个来源参数时，明确使用此前已归档的识别预测，适合单独检查定位接入。
& $researchPython -m fp_research localize --system bank --stratum 1 --version '_MultipleBugs_.NOB_1.ID_14' --run-name locate-one
```

定位派生文件保存到 `案例/.fp_research/localization/运行名/`，汇总保存到 `experimental_results/localization_runs/运行名/`。流程依次为排名、逐位置评价、独立复算、中文汇总。`--stage rank/evaluate/verify/summary` 可单步执行；`--list` 只列案例。新运行不写入历史 `statement_localization` 或 `statement_localization_report_relabel` 目录。

此接入仍保存CSV和压缩排名文件，不调用原版生成xlsx的外层入口。若你人工运行原版VARCOP，请选择本次独立修正报告，并重建对应特征组合和切片，避免使用旧结果。

## 检查

```powershell
& $researchPython -m unittest discover -s tests -v
& $researchPython -m fp_research archive-check --manifest 'experimental_results\statement_localization_report_relabel\archive_manifest.json'
```

第三方比较方法检查需要本机CLAP源码和实验依赖。只检查主模型时，可以运行 `python -m unittest tests.test_model tests.test_detection_cli -v`。
