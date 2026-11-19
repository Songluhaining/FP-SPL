# 2026-11-19 配置报告修正协议

之前整配置剔除的定位结果不符合用户要求，仅保留为历史协议，不作为本轮结果。

1. 在各案例 `.fp_research/statement_localization_report_relabel/方法/` 新增 `config.report.csv`。只把预测假通过配置最后一列由 `__PASSED__` 改成 `__FAILED__`，其余字节、配置行与特征不变。
2. 替换官方公共报告解析函数和已导入该函数的两个模块，保证特征组合计算、基础特征故障判断都读取对应方法的新报告。绝不剔除配置。
3. 严格保持原程序的覆盖评分逻辑：逐测试成败与覆盖不改，产品报告修正不伪造失败测试。原程序的产品级覆盖评分依靠实际失败覆盖文件，与读取报告的环节分开；本轮不额外修改它。
4. 新标失败的产品仍参与特征组合计算。没有实际失败覆盖时不伪造切片证据，单独登记，切片仅使用已有实际失败覆盖的产品。保存完整组合日志和实际进入切片的日志。
5. 引用上一轮相同的联合推断与两种基线折外预测，不重训、不调阈值。新输出目录与历史结果隔离，不复制数据集。
6. 先核验银行账户73个单缺陷案例；官方枚举结果一致、原始切片工具候选集合一致、未修正六公式排名全部复现后，才扩展1054个案例。图处理与压缩系统多缺陷仍暂缓。

核验命令（仓库根目录，Python 为本机 `D:/SoftInstall/python.exe`）：

```powershell
python -m unittest fp_research.checks.test_report_relabel -v
python -m fp_research.report_relabel_study all --system bank --stratum 1 --workers 3
python -m fp_research.checks.report_relabel_reference
python -m fp_research.report_relabel_study all --workers 3
python -m fp_research.checks.verify_report_relabel
```

第一阶段实测：3项小样例核验通过；73个单缺陷案例、2482个配置全部保留，未修正六公式的完整及筛选排名复现；主公式联合修正平均检查比例从0.047382变为0.047021。完整结果只能在全部案例运行及独立统计核验通过后发布。
