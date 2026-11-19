# 项目数据与实验约定

- 活动代码分为data、detection、baselines、localization、experiments、evaluation；新功能放到对应层，不恢复已删除的历史批处理脚本。
- 推荐入口为 `python -m fp_research detect/baseline/localize/archive-check`；上一版manual_joint命令仅作兼容。
- 数据路径只在configs/datasets.json维护，外部工具路径只在configs/tools.json维护。银行账户、图处理、压缩在D:/dataset/whn；邮件、电梯、考试数据库在F:/datasets。
- 银行账户、邮件、电梯、考试数据库包含单、双、三缺陷；图处理、压缩双三缺陷仍暂缓，不自动恢复。
- 原始源码、测试、报告及覆盖只读；案例派生文件写入各案例.fp_research。除用户明确要求导出的新报告外，不在原始根目录写文件，不复制整套数据集。
- 主方法逐案例独立初始化，不读取变异日志或真假通过评价标签；目录中的缺陷数不告知推断模型。比较方法单独记录训练与评价版本。
- 假通过修正只生成独立报告并改最后一列，不剔除配置，不伪造失败测试或修改原覆盖。
- experimental_results中的历史文件不覆盖、不改写。旧源码摘要由对应Git提交复核，不用新路径冒充旧源码；重构前标签为archive/pre-refactor-20261119。
- 每次重要修改完成相应核验后提交，保留逐配置或逐位置明细及独立统计凭据。
