package ppfl;

/**
 * SPL 特征节点：
 * - 继承 Node，只是语义上表示一个“特征”；
 * - 本身仍然是二值变量：true = 该特征整体正确，false = 该特征存在错误；
 * - isStmt = false，不参与语句相关的特殊逻辑。
 */
public class FeatureNode extends Node {

    private final String featureName;

    public FeatureNode(String featureName) {
        super(featureName);
        this.featureName = featureName;
        this.isStmt = false; // 明确标记：不是语句节点
    }

    public String getFeatureName() {
        return featureName;
    }

    @Override
    public String getName() {
        // 对特征节点，不拼接 testname，直接用特征名
        return this.featureName;
    }

    @Override
    public String getPrintName() {
        return this.featureName;
    }

    @Override
    public void print(MyWriter lgr, String prefix) {
        if (this.obs) {
            lgr.writeln("%s%s(Feature) observed = %b",
                    prefix, this.featureName, this.obsvalue);
        } else {
            lgr.writeln("%s%s(Feature)", prefix, this.featureName);
        }
    }

    @Override
    public void print(MyWriter lgr) {
        if (this.obs) {
            lgr.writeln("%s(Feature) observed = %b",
                    this.featureName, this.obsvalue);
        } else {
            lgr.writeln("%s(Feature)", this.featureName);
        }
    }

    @Override
    public void print(String prefix) {
        if (this.obs) {
            printLogger.writeln("%s%s(Feature) observed = %b",
                    prefix, this.featureName, this.obsvalue);
        } else {
            printLogger.writeln("%s%s(Feature)", prefix, this.featureName);
        }
    }

    @Override
    public void print() {
        if (this.obs) {
            printLogger.writeln("%s(Feature) observed = %b",
                    this.featureName, this.obsvalue);
        } else {
            printLogger.writeln("%s(Feature)", this.featureName);
        }
    }
}
