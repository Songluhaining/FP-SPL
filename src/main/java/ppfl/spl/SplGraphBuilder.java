package ppfl.spl;

import ppfl.ByteCodeGraph;
import ppfl.WriterUtils;

import java.io.File;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.w3c.dom.Element;

import ppfl.StmtNode;
import ppfl.FactorNode;
import ppfl.FeatureNode;
import ppfl.Edge;
import ppfl.Node;

/**
 * SPL 场景下的图构建入口（簇级别）：
 * - 支持 main(args) 传入多个 variantRoot；
 * - 使用 coverage XML 做 Feature 映射；
 * - 将 mytrace/run 日志重写到特征空间；
 * - 针对一个聚类簇（clusterId）、只使用该簇中的失败测试；
 * - 在全局因子图上跑一次 Belief Propagation；
 * - 额外：从 mutant 根目录下的 stmt_sbfl_priors.txt 读取 SBFL 语句先验，挂到 StmtNode 上。
 */
public class SplGraphBuilder {

    /* ===================== 1. Feature 映射器 ===================== */

    /**
     * (类名, 源码行号) 所对应的特征位置信息。
     */
    public static class FeatureLocation {
        public final String featureClass; // 如 "Base.ElevatorSystem.Elevator"
        public final int featureLine;     // 如 34
        public final String type;         // "stmt" / "method" 等，用于选择优先级

        public FeatureLocation(String featureClass, int featureLine, String type) {
            this.featureClass = featureClass;
            this.featureLine = featureLine;
            this.type = type;
        }
    }

    /**
     * 从 coverage/{spectrum_failed_coverage.xml, spectrum_passed_coverage.xml}
     * 中读取 (类名 + 源码行) → (特征类 + 特征行) 的映射，并提供对 lineinfo 的重写能力。
     */
    public static class FeatureLocationMapper {

        // key: fileClassName + ":" + srcLineNum
        private final Map<String, FeatureLocation> mapping = new HashMap<>();

        /**
         * 构造函数：给定一个 variantRoot，自动去找 coverage 目录下面的 coverage XML。
         */
        public FeatureLocationMapper(File variantRoot) {
            File coverageDir = new File(variantRoot, "coverage");
            if (!coverageDir.exists() || !coverageDir.isDirectory()) {
                System.err.println("[FeatureMapper][WARN] coverage dir not found: " + coverageDir);
                return;
            }

            File failedXml = new File(coverageDir, "spectrum_failed_coverage.xml");
            File passedXml = new File(coverageDir, "spectrum_passed_coverage.xml");

            loadOneCoverageXml(failedXml);
            loadOneCoverageXml(passedXml);

            System.out.println("[FeatureMapper] loaded mappings for variant " +
                    variantRoot.getName() + ", size=" + mapping.size());
        }

        private void loadOneCoverageXml(File xmlFile) {
            if (xmlFile == null || !xmlFile.isFile()) {
                return;
            }

            try {
                javax.xml.parsers.DocumentBuilderFactory factory =
                        javax.xml.parsers.DocumentBuilderFactory.newInstance();
                factory.setIgnoringComments(true);
                factory.setIgnoringElementContentWhitespace(true);

                javax.xml.parsers.DocumentBuilder builder = factory.newDocumentBuilder();
                org.w3c.dom.Document doc = builder.parse(xmlFile);
                doc.getDocumentElement().normalize();

                // 所有 <file> 节点统一处理，父节点里会有 <package name="...">
                org.w3c.dom.NodeList fileNodes = doc.getElementsByTagName("file");
                for (int i = 0; i < fileNodes.getLength(); i++) {
                    org.w3c.dom.Node fNode = fileNodes.item(i);
                    if (!(fNode instanceof org.w3c.dom.Element)) {
                        continue;
                    }
                    org.w3c.dom.Element fileElem = (org.w3c.dom.Element) fNode;

                    // 1) 读出 file 的短类名和路径
                    String shortName = fileElem.getAttribute("name");   // 例如 "Transaction" 或 "Base.ElevatorSystem.Elevator"
                    String path      = fileElem.getAttribute("path");   // 例如 "Transaction.java" 或 "Base/ElevatorSystem/Elevator.java"

                    if (shortName == null || shortName.isEmpty()) {
                        continue;
                    }

                    shortName = shortName.trim();

                    // 2) 构造尽可能完整的类名
                    String classNameWithPkg = shortName;

                    // 2.1 如果 shortName 本身已经包含 '.'，认为已经是完整类名（Elevator FH 常见）
                    if (!shortName.contains(".")) {
                        // 2.2 尝试从父 <package name="..."> 取包名
                        String pkgName = null;
                        org.w3c.dom.Node parent = fileElem.getParentNode();
                        while (parent != null) {
                            if (parent instanceof org.w3c.dom.Element) {
                                org.w3c.dom.Element e = (org.w3c.dom.Element) parent;
                                if ("package".equals(e.getTagName())) {
                                    pkgName = e.getAttribute("name");
                                    if (pkgName != null && !pkgName.isEmpty()) {
                                        pkgName = pkgName.trim();
                                    }
                                    break;
                                }
                            }
                            parent = parent.getParentNode();
                        }

                        if (pkgName != null && !pkgName.isEmpty()) {
                            // BankAccountTP: package="main", name="Transaction" -> "main.Transaction"
                            classNameWithPkg = pkgName + "." + shortName;
                        } else if (path != null && !path.isEmpty()) {
                            // 2.3 没有 package 信息时，尝试根据 path 推导
                            // 例如 "Base/ElevatorSystem/Elevator.java" -> "Base.ElevatorSystem.Elevator"
                            String tmp = path.trim()
                                    .replace('\\', '/');          // 统一分隔符
                            if (tmp.endsWith(".java")) {
                                tmp = tmp.substring(0, tmp.length() - ".java".length());
                            }
                            tmp = tmp.replace('/', '.');
                            if (!tmp.isEmpty()) {
                                classNameWithPkg = tmp;
                            }
                        }
                    }

                    // 3) 遍历该文件下所有 <line> 节点
                    org.w3c.dom.NodeList lineNodes = fileElem.getElementsByTagName("line");
                    for (int j = 0; j < lineNodes.getLength(); j++) {
                        org.w3c.dom.Node lNode = lineNodes.item(j);
                        if (!(lNode instanceof org.w3c.dom.Element)) {
                            continue;
                        }
                        org.w3c.dom.Element lineElem = (org.w3c.dom.Element) lNode;

                        String numStr          = lineElem.getAttribute("num");          // 源码行号
                        String type            = lineElem.getAttribute("type");         // "stmt" / "method" 等
                        String featureClass    = lineElem.getAttribute("featureClass"); // 目标特征类，例如 "Transaction.Transaction"
                        String featureLineStr  = lineElem.getAttribute("featureLineNum");

                        if (numStr == null || numStr.isEmpty()) {
                            continue;
                        }
                        if (featureClass == null || featureClass.isEmpty()
                                || featureLineStr == null || featureLineStr.isEmpty()) {
                            // 没有特征空间信息，跳过
                            continue;
                        }

                        numStr         = numStr.trim();
                        featureClass   = featureClass.trim();
                        featureLineStr = featureLineStr.trim();
                        if (type != null) {
                            type = type.trim();
                        }

                        int srcLine;
                        int featureLine;
                        try {
                            srcLine = Integer.parseInt(numStr);
                            featureLine = Integer.parseInt(featureLineStr);
                        } catch (NumberFormatException e) {
                            // 非法数字，跳过这一行
                            continue;
                        }

                        // 4) 构造 FeatureLocation
                        FeatureLocation newLoc = new FeatureLocation(featureClass, featureLine, type);

                        // 5) 构造多种 key，以适配不同数据集的类名写法
                        //    key = <类名>:<源码行号>
                        String keyShort = shortName + ":" + srcLine;            // "Transaction:13"
                        String keyFull  = classNameWithPkg + ":" + srcLine;     // "main.Transaction:13" 或 "Base.ElevatorSystem.Elevator:42"

                        // 可选：再补一份只用 path 推的类名（如果和上面不同）
                        String keyFromPath = null;
                        if (path != null && !path.isEmpty()) {
                            String tmp = path.trim().replace('\\', '/');
                            if (tmp.endsWith(".java")) {
                                tmp = tmp.substring(0, tmp.length() - ".java".length());
                            }
                            tmp = tmp.replace('/', '.'); // "Base.ElevatorSystem.Elevator"
                            if (!tmp.isEmpty()) {
                                keyFromPath = tmp + ":" + srcLine;
                            }
                        }

                        // 6) 写入 mapping，冲突时优先保留 "stmt" 行
                        // ---- keyShort ----
                        FeatureLocation old = mapping.get(keyShort);
                        if (old == null) {
                            mapping.put(keyShort, newLoc);
                        } else {
                            boolean oldIsStmt = "stmt".equals(old.type);
                            boolean newIsStmt = "stmt".equals(newLoc.type);
                            if (!oldIsStmt && newIsStmt) {
                                mapping.put(keyShort, newLoc);
                            }
                        }

                        // ---- keyFull ----
                        if (!keyFull.equals(keyShort)) {
                            old = mapping.get(keyFull);
                            if (old == null) {
                                mapping.put(keyFull, newLoc);
                            } else {
                                boolean oldIsStmt = "stmt".equals(old.type);
                                boolean newIsStmt = "stmt".equals(newLoc.type);
                                if (!oldIsStmt && newIsStmt) {
                                    mapping.put(keyFull, newLoc);
                                }
                            }
                        }

                        // ---- keyFromPath（可选，但很鲁棒）----
                        if (keyFromPath != null
                                && !keyFromPath.equals(keyShort)
                                && !keyFromPath.equals(keyFull)) {
                            old = mapping.get(keyFromPath);
                            if (old == null) {
                                mapping.put(keyFromPath, newLoc);
                            } else {
                                boolean oldIsStmt = "stmt".equals(old.type);
                                boolean newIsStmt = "stmt".equals(newLoc.type);
                                if (!oldIsStmt && newIsStmt) {
                                    mapping.put(keyFromPath, newLoc);
                                }
                            }
                        }
                    }
                }

            } catch (Exception e) {
                // 解析 XML 出错时，不让整个流程挂掉，只打印一下方便调试
                e.printStackTrace();
            }
        }

        /**
         * 给定原始 lineinfo 字符串，尝试根据映射转换到 Feature 空间。
         * lineinfo 形如：
         *   ElevatorSystem.Elevator#<init>#(LElevatorSystem/Environment;Z)V#58#21
         *
         * 若在映射表中找到 (className, srcLine)，则将 className 和 srcLine 换成
         * featureClass 和 featureLineNum，其余保持不变。
         */
        public String mapLineInfoIfPossible(String lineInfo) {
            if (lineInfo == null || lineInfo.isEmpty()) {
                return lineInfo;
            }
            String[] parts = lineInfo.split("#");
            if (parts.length < 5) {
                return lineInfo;
            }

            String className = parts[0];
            String method = parts[1];
            String desc = parts[2];
            String srcLineStr = parts[3];
            String instIdx = parts[4];

            int srcLine;
            try {
                srcLine = Integer.parseInt(srcLineStr);
            } catch (NumberFormatException e) {
                return lineInfo;
            }

            String key = className + ":" + srcLine;
            FeatureLocation loc = mapping.get(key);
            if (loc == null) {
                return lineInfo;
            }

            String featureClass = loc.featureClass;
            int featureLine = loc.featureLine;

            return featureClass + "#" + method + "#" + desc + "#" + featureLine + "#" + instIdx;
        }

        /**
         * 对一整行日志进行处理：若存在 lineinfo=... 则只替换该字段的值。
         */
        public String rewriteLineWithFeature(String line) {
            if (line == null) return null;
            int idx = line.indexOf("lineinfo=");
            if (idx < 0) {
                return line;
            }
            int start = idx + "lineinfo=".length();
            // lineinfo 后面到下一逗号(,) 或 行尾为值
            int end = line.indexOf(",", start);
            if (end < 0) {
                end = line.length();
            }
            String oldLineInfo = line.substring(start, end).trim();
            String newLineInfo = mapLineInfoIfPossible(oldLineInfo);
            if (newLineInfo.equals(oldLineInfo)) {
                return line;
            }
            // 保留原有前后结构，只替换中间 lineinfo 值
            StringBuilder sb = new StringBuilder();
            sb.append(line, 0, start);
            sb.append(newLineInfo);
            sb.append(line.substring(end));
            return sb.toString();
        }
    }

    /* ===================== 2. 测试初始化（按簇过滤） ===================== */

    /**
     * 从 runFolder 里的 *.log 推断测试名，并只注册当前簇允许的测试。
     */
    private static void initSplTestsFromRunLogs(
            ByteCodeGraph pgraph,
            String runFolder,
            String variantName,
            Set<String> allowedTestsForVariant) {

        File folder = new File(runFolder);
        if (!folder.exists() || !folder.isDirectory()) {
            System.err.println("[SPL][WARN] run folder not found: " + runFolder);
            return;
        }

        File[] logs = folder.listFiles((dir, name) -> name.endsWith(".log"));
        if (logs == null || logs.length == 0) {
            System.err.println("[SPL][WARN] no .log files found in run folder: " + runFolder);
            return;
        }

        for (File f : logs) {
            String fname = f.getName();
            if (!fname.endsWith(".log")) {
                continue;
            }

            String base = fname.substring(0, fname.length() - ".log".length());
            int dotIdx = base.indexOf('.');
            String testName;
            if (dotIdx <= 0 || dotIdx == base.length() - 1) {
                // default-package class (no package, e.g. TankWar): empty package, keep "::"
                testName = "::" + base;             // "::Maler_test41"
            } else {
                String cls  = base.substring(0, dotIdx);  // ElevatorSystem
                String meth = base.substring(dotIdx + 1); // Elevator_test53
                testName = cls + "::" + meth;
            }

            if (allowedTestsForVariant != null && !allowedTestsForVariant.contains(testName)) {
                System.out.println("[SPL][Cluster] skip test (not in this cluster): variant=" +
                        variantName + ", test=" + testName);
                continue;
            }

            System.out.println("[SPL] register test from logfile: " + fname + " -> " + testName);

            pgraph.d4jMethodNames.add(testName);
            // 这里仍假定 run/*.log 都是 FAIL；如果有 PASS，可以根据 oracle 再做区分
            pgraph.d4jTriggerTestNames.add(testName);
        }
    }

    /**
     * 从 tests-oracle.csv 初始化测试集，并只保留当前簇允许的测试。
     */
    private static void initSplTestsFromOracleFile(
            ByteCodeGraph pgraph,
            String oracleFilePath,
            String variantName,
            Set<String> allowedTestsForVariant) {

        File f = new File(oracleFilePath);
        if (!f.exists()) {
            System.err.println("[SPL][WARN] oracle file not found: " + oracleFilePath);
            return;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;

                String[] parts = line.split(",");
                if (parts.length < 2) {
                    System.err.println("[SPL][WARN] bad oracle line: " + line);
                    continue;
                }
                String testName = parts[0].trim();         // 约定为 Cls::meth
                String status   = parts[1].trim().toUpperCase();

                if (allowedTestsForVariant != null && !allowedTestsForVariant.contains(testName)) {
                    continue;
                }

                pgraph.d4jMethodNames.add(testName);
                if (status.equals("FAIL")) {
                    pgraph.d4jTriggerTestNames.add(testName);
                }
            }
        } catch (IOException e) {
            System.err.println("[SPL][ERR] fail to read oracle file: " + oracleFilePath);
            e.printStackTrace();
        }

        System.out.println("[SPL] tests from oracle (clustered): total="
                + pgraph.d4jMethodNames.size()
                + ", failing=" + pgraph.d4jTriggerTestNames.size());
    }

    /**
     * 将 cluster_*.txt 中的测试名规范化为 run 日志中使用的测试名
     * 例如： "ElevatorSystem::Elevator_ESTest.test00"
     *   ->   "ElevatorSystem::Elevator_test00"
     */
    private static String normalizeClusterTestName(String raw) {
        if (raw == null || raw.isEmpty()) {
            return raw;
        }

        int sep = raw.indexOf("::");
        if (sep < 0) {
            // 没有 "::" 的奇怪格式，先保持原样
            return raw;
        }

        String cls = raw.substring(0, sep);         // "ElevatorSystem"
        String methodPart = raw.substring(sep + 2); // "Elevator_ESTest.test00"

        // 如果右半部分本身就不含 '.'，例如已经是 "Elevator_test00"，说明已经是规范形式
        if (!methodPart.contains(".")) {
            return raw;
        }

        // 处理 "Elevator_ESTest.test00" 这种模式
        String[] parts = methodPart.split("\\.");
        if (parts.length == 2) {
            String estestClass = parts[0]; // "Elevator_ESTest"
            String testMethod  = parts[1]; // "test00"

            if (estestClass.endsWith("_ESTest") && testMethod.startsWith("test")) {
                // 去掉 "_ESTest"，把 "test00" 拼到后面，得到 "Elevator_test00"
                String baseName  = estestClass.substring(0, estestClass.length() - "_ESTest".length()); // "Elevator"
                String newMethod = baseName + "_test" + testMethod.substring("test".length());          // "Elevator_test00"
                String normalized = cls + "::" + newMethod;

                System.out.println("[SPL][Cluster] normalize test: " + raw + " -> " + normalized);
                return normalized;
            }
        }

        // 其它不认识的格式，先保持原样
        return raw;
    }

    /* ===================== 3. 日志重写工具 ===================== */

    /**
     * 将 variant 下的 mytrace/*.source.log 映射到 Feature 空间，
     * 写入 trace/logs/mytrace_featured 目录，返回该目录 File。
     */
    private static File rewriteSourceLogsForVariant(File variantRoot, FeatureLocationMapper mapper) {
        File srcDir = new File(variantRoot, "trace/logs/mytrace");
        if (!srcDir.exists() || !srcDir.isDirectory()) {
            System.err.println("[SPL][WARN] mytrace dir not found: " + srcDir);
            return srcDir; // 返回原目录，后面解析时会自动跳过
        }

        File outDir = new File(variantRoot, "trace/logs/mytrace_featured");
        if (!outDir.exists() && !outDir.mkdirs()) {
            System.err.println("[SPL][ERR] cannot create mytrace_featured: " + outDir);
            return srcDir;
        }

        File[] files = srcDir.listFiles();
        if (files == null) {
            return outDir;
        }

        for (File inFile : files) {
            String fname = inFile.getName();
            if (!fname.endsWith(".source.log") || fname.endsWith("traced.source.log")) {
                // traced.source.log 通常是合并文件，先保持原样不处理
                copyFileIfNeeded(inFile, new File(outDir, fname));
                continue;
            }

            File outFile = new File(outDir, fname);
            try (
                    BufferedReader br = new BufferedReader(new FileReader(inFile));
                    PrintWriter pw = new PrintWriter(new FileWriter(outFile))
            ) {
                String line;
                while ((line = br.readLine()) != null) {
                    String newline = mapper.rewriteLineWithFeature(line);
                    pw.println(newline);
                }
            } catch (IOException e) {
                System.err.println("[SPL][ERR] rewrite source log failed: " + inFile);
                e.printStackTrace();
                // 出问题时直接复制原文件到 _featured，避免缺文件
                copyFileIfNeeded(inFile, outFile);
            }
        }

        return outDir;
    }

    /**
     * 将 variant 下的 run/*.log 映射到 Feature 空间，
     * 写入 trace/logs/run_featured 目录，返回该目录 File。
     */
    private static File rewriteRunLogsForVariant(File variantRoot, FeatureLocationMapper mapper) {
        File runDir = new File(variantRoot, "trace/logs/run");
        if (!runDir.exists() || !runDir.isDirectory()) {
            System.err.println("[SPL][WARN] run dir not found: " + runDir);
            return runDir;
        }

        File outDir = new File(variantRoot, "trace/logs/run_featured");
        if (!outDir.exists() && !outDir.mkdirs()) {
            System.err.println("[SPL][ERR] cannot create run_featured: " + outDir);
            return runDir;
        }

        File[] files = runDir.listFiles((dir, name) -> name.endsWith(".log"));
        if (files == null) {
            return outDir;
        }

        for (File inFile : files) {
            String fname = inFile.getName();
            File outFile = new File(outDir, fname);
            try (
                    BufferedReader br = new BufferedReader(new FileReader(inFile));
                    PrintWriter pw = new PrintWriter(new FileWriter(outFile))
            ) {
                String line;
                while ((line = br.readLine()) != null) {
                    String newline = mapper.rewriteLineWithFeature(line);
                    pw.println(newline);
                }
            } catch (IOException e) {
                System.err.println("[SPL][ERR] rewrite run log failed: " + inFile);
                e.printStackTrace();
                copyFileIfNeeded(inFile, outFile);
            }
        }

        return outDir;
    }

    private static void copyFileIfNeeded(File src, File dst) {
        if (dst.exists()) {
            return;
        }
        try (
                BufferedReader br = new BufferedReader(new FileReader(src));
                PrintWriter pw = new PrintWriter(new FileWriter(dst))
        ) {
            String line;
            while ((line = br.readLine()) != null) {
                pw.println(line);
            }
        } catch (IOException e) {
            System.err.println("[SPL][ERR] copy file failed: " + src + " -> " + dst);
            e.printStackTrace();
        }
    }

    /* ===================== 2.x 语句 / 特征工具 ===================== */

    /**
     * 根据 StmtNode 的 classMethod 提取“顶层特征名”。
     *
     * 当前 name 形如：
     *   Base.ElevatorSystem.Elevator:<init>:(I)V#58#21
     * StmtNode.getClassMethod() 返回：
     *   "Base.ElevatorSystem.Elevator"
     *
     * 我们按你的设定，只取第一个 '.' 之前的前缀作为特征：
     *   "Base.ElevatorSystem.Elevator" -> "Base"
     *   "Weight.ElevatorSystem.Environment" -> "Weight"
     */
    private static String extractFeatureNameFromStmt(StmtNode stmt) {
        if (stmt == null) {
            return null;
        }
        String featureClass = stmt.getClassMethod(); // 如 "Base.ElevatorSystem.Elevator"
        if (featureClass == null || featureClass.isEmpty()) {
            return null;
        }

        int dotIdx = featureClass.indexOf('.');
        if (dotIdx <= 0) {
            // 没有 '.'，直接把整个字符串当成一个特征
            return featureClass;
        }
        return featureClass.substring(0, dotIdx); // 只取 "Base"
    }

    /**
     * 为语句节点构造一个 key，用来和 SBFL 文件对应：
     *   key = featureClass + "." + lineNumber
     * 其中 featureClass 形如 "Weight.ElevatorSystem.Elevator"
     * 行号从 stmt.getName() 中的 "...#22#9" 解析出 "22"。
     */
    private static String buildStmtKey(StmtNode stmt) {
        if (stmt == null) return null;
        String klass = stmt.getClassMethod();  // e.g. "Weight.ElevatorSystem.Elevator"
        if (klass == null || klass.isEmpty()) return null;

        String fullName = stmt.getName();      // e.g. "Weight.ElevatorSystem.Elevator:...Z#22#9"
        if (fullName == null) return null;

        int lastHash = fullName.lastIndexOf('#');
        if (lastHash < 0) return null;

        String beforeLast = fullName.substring(0, lastHash); // "...Z#22"
        int secondHash = beforeLast.lastIndexOf('#');
        if (secondHash < 0 || secondHash == beforeLast.length() - 1) {
            return null;
        }

        String lineStr = beforeLast.substring(secondHash + 1);
        int line;
        try {
            line = Integer.parseInt(lineStr);
        } catch (NumberFormatException e) {
            return null;
        }

        return klass + "." + line;  // e.g. "Weight.ElevatorSystem.Elevator.22"
    }

    /**
     * 统一将“顶层特征名”转换成图中 FeatureNode 的 key/name。
     */
    private static String toFeatureNodeKey(String featureName) {
        return featureName;
    }

    /**
     * 在 pgraph.nodes（List<Node>）中，根据顶层特征名查找对应的 FeatureNode。
     */
    private static Node findFeatureNode(ByteCodeGraph pgraph, String featureName) {
        String nodeName = toFeatureNodeKey(featureName);
        for (Node n : pgraph.nodes) {
            if (nodeName.equals(n.getName())) {
                return n;
            }
        }
        return null;
    }

    /**
     * 在全局 ByteCodeGraph 上，为每个语句节点挂接一个“顶层特征节点”，并构建
     * “语句–特征”二元因子。
     */
    private static void addFeatureFactors(ByteCodeGraph pgraph) {
        if (pgraph == null || pgraph.stmts == null || pgraph.stmts.isEmpty()) {
            System.err.println("[SPL][Feature] no stmts in graph, skip feature factor construction.");
            return;
        }

        // key: 顶层特征名，如 "Base"；value: FeatureNode 实例
        Map<String, FeatureNode> featureNodes = new HashMap<>();

        int addedFeatureNodes = 0;
        int addedFeatureFactors = 0;

        // pgraph.stmts 是 List<StmtNode>
        for (StmtNode stmt : pgraph.stmts) {
            String topFeatureName = extractFeatureNameFromStmt(stmt);
            if (topFeatureName == null || topFeatureName.isEmpty()) {
                continue;
            }

            // 1. 获取或创建 FeatureNode（一个顶层特征一个节点）
            FeatureNode fnode = featureNodes.get(topFeatureName);
            if (fnode == null) {
                fnode = new FeatureNode(topFeatureName);
                featureNodes.put(topFeatureName, fnode);

                // 加入全局变量节点列表（ByteCodeGraph.nodes 是 List<Node>）
                pgraph.nodes.add(fnode);
                addedFeatureNodes++;
            }

            // 2. 为语句节点和特征节点各自创建一条 Edge
            Edge sedge = new Edge();
            sedge.setnode(stmt);
            stmt.add_edge(sedge);

            Edge fedge = new Edge();
            fedge.setnode(fnode);
            fnode.add_edge(fedge);

            // 3. 创建语句–特征因子
            FactorNode sfFactor = new FactorNode(stmt, fnode, sedge, fedge);

            // 4. 将因子回填到 Edge
            sedge.setfactor(sfFactor);
            fedge.setfactor(sfFactor);

            // 5. 加入全局因子列表
            pgraph.factornodes.add(sfFactor);
            addedFeatureFactors++;
        }

        System.out.println("[SPL][Feature] added FeatureNodes=" + addedFeatureNodes
                + ", FeatureFactors=" + addedFeatureFactors);
        System.out.println("[SPL][Feature][DEBUG] now vars=" + pgraph.nodes.size()
                + ", factors=" + pgraph.factornodes.size());
    }

    private static String deduceGlobalName(String firstVariantRoot) {
        File vr = new File(firstVariantRoot);
        String fallback = vr.getName();

        File parent = vr.getParentFile(); // 通常是 variants
        if (parent != null) {
            File gparent = parent.getParentFile(); // 通常是 mutant 目录
            if (gparent != null) {
                return gparent.getName();
            }
        }
        return fallback;
    }

    /**
     * 为给定的“可疑顶层特征”设置先验因子。
     */
    private static void addSuspiciousFeaturePriors(
            ByteCodeGraph pgraph,
            Map<String, Double> suspiciousFeaturePriors) {

        if (suspiciousFeaturePriors == null || suspiciousFeaturePriors.isEmpty()) {
            System.out.println("[SPL] no suspicious single features, skip priors.");
            return;
        }

        int added = 0;

        for (Map.Entry<String, Double> e : suspiciousFeaturePriors.entrySet()) {
            String featureName = e.getKey();      // 如 "Base"
            double pCorrect    = e.getValue();   // P(f=true)

            Node fnode = findFeatureNode(pgraph, featureName);
            if (fnode == null) {
                System.err.println("[SPL][WARN] suspicious feature node not found: " + toFeatureNodeKey(featureName));
                continue;
            }

            // 为该特征挂一个一元因子，编码 P(f=true)=pCorrect
            Edge edge = new Edge();
            edge.setnode(fnode);
            fnode.add_edge(edge);

            FactorNode fn = new FactorNode(fnode, edge, pCorrect);
            edge.setfactor(fn);

            pgraph.factornodes.add(fn);
            added++;
        }

        System.out.println("[SPL][Feature] added suspicious feature priors=" + added);
    }

    /**
     * 从 mutant 根目录下的 stmt_sbfl_priors.txt 读取语句先验：
     * 每行：<stmtKey> <suspiciousness> 或 <stmtKey>,<suspiciousness>
     * 其中 stmtKey = "Feature.ClassName.line"，例如 "Weight.ElevatorSystem.Elevator.22"
     * suspiciousness ∈ [0,1] 越大越可疑。
     * 我们映射为语句为 true（正确）的先验概率：pCorrect = 1 - suspiciousness，并做轻微平滑。
     */
    private static Map<String, Map<String, Double>> loadStmtSbflPriorsMulti(String path) {
        Map<String, Map<String, Double>> byMetric = new HashMap<>();

        File f = new File(path);
        if (!f.exists()) {
            System.err.println("[SPL][StmtPrior][WARN] stmt prior file not found: " + path);
            return byMetric;
        }

        System.out.println("[SPL][StmtPrior] loading multi-metric stmt priors from: " + f.getAbsolutePath());

        int badLines = 0;
        int totalPairs = 0;

        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;

                // tokens: key + metric=value + metric=value ...
                String[] parts = line.split("\\s+");
                if (parts.length < 2) continue;

                String key = parts[0].trim();

                for (int i = 1; i < parts.length; i++) {
                    String tok = parts[i].trim();
                    if (tok.isEmpty()) continue;

                    int eq = tok.indexOf('=');
                    if (eq <= 0 || eq == tok.length() - 1) {
                        // 不符合 metric=value，跳过
                        badLines++;
                        continue;
                    }

                    String metric = tok.substring(0, eq).trim();
                    String valStr = tok.substring(eq + 1).trim();

                    double susp;
                    try {
                        susp = Double.parseDouble(valStr);
                    } catch (NumberFormatException e) {
                        System.out.println("[SPL][StmtPrior][WARN] bad susp value: " + line);
                        badLines++;
                        break; // 这一行基本废了，退出 token 循环
                    }

                    // susp 越大越可疑 -> pCorrect 越小
                    double pCorrect = 1.0 - susp;
                    double eps = 1e-3;
                    if (pCorrect < eps) pCorrect = eps;
                    if (pCorrect > 1.0 - eps) pCorrect = 1.0 - eps;

                    byMetric.computeIfAbsent(metric, k -> new HashMap<>()).put(key, pCorrect);
                    totalPairs++;
                }
            }
        } catch (IOException e) {
            System.err.println("[SPL][StmtPrior][ERR] fail to read stmt prior file: " + path);
            e.printStackTrace();
        }

        System.out.println("[SPL][StmtPrior] loaded metrics: " + byMetric.keySet());
        System.out.println("[SPL][StmtPrior] total stmt prior pairs = " + totalPairs + ", badLines=" + badLines);
        return byMetric;
    }

    /**
     * 从所有 *_featured 的 dyn.log 中收集“动态执行过”的类名。
     *
     * 例如一行 dyn.log 里有:
     *   ###EmailSystem.Client::someMethod
     * 则提取类名 "EmailSystem.Client" 加入 dynamicClasses 集合。
     */
    private static Set<String> collectDynamicClasses(File[] mytraceFeaturedDirs) {
        Set<String> dynamicClasses = new HashSet<>();

        if (mytraceFeaturedDirs == null) {
            return dynamicClasses;
        }

        for (File mytraceDir : mytraceFeaturedDirs) {
            if (mytraceDir == null || !mytraceDir.exists() || !mytraceDir.isDirectory()) {
                continue;
            }
            File[] fs = mytraceDir.listFiles();
            if (fs == null || fs.length == 0) {
                continue;
            }

            for (File f : fs) {
                String fname = f.getName();
                // 只看 dyn.log
                if (!fname.endsWith(".dyn.log") || fname.endsWith("traced.dyn.log")) {
                    continue;
                }

                try (BufferedReader br = new BufferedReader(new FileReader(f))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        line = line.trim();
                        if (line.isEmpty()) continue;
                        if (!line.startsWith("###")) continue;

                        // 形如: ###EmailSystem.Client::someMethod
                        String body = line.substring(3).trim();
                        int idx = body.indexOf("::");
                        if (idx <= 0) {
                            continue;
                        }
                        String cls = body.substring(0, idx).trim(); // "EmailSystem.Client"
                        if (!cls.isEmpty()) {
                            dynamicClasses.add(cls);
                        }
                    }
                } catch (IOException e) {
                    System.err.println("[SPL][WARN] fail to read dyn log: " + f.getAbsolutePath());
                    e.printStackTrace();
                }
            }
        }

        System.out.println("[SPL][Dynamic] collected classes: " + dynamicClasses.size());
        return dynamicClasses;
    }

    /**
     * 为匹配到的语句节点加上一元先验因子（来自 SBFL）。
     */
    private static List<FactorNode> addStmtSbflPriors(ByteCodeGraph pgraph, Map<String, Double> stmtPriors) {
        List<FactorNode> addedFactors = new ArrayList<>();
        if (stmtPriors == null || stmtPriors.isEmpty()) {
            System.out.println("[SPL][StmtPrior] no stmt priors, skip.");
            return addedFactors;
        }

        int added = 0;
        for (StmtNode stmt : pgraph.stmts) {
            String key = buildStmtKey(stmt); // 你原来怎么构造 stmtKey 就保持不变
            if (key == null) continue;

            Double pCorrect = stmtPriors.get(key);
            if (pCorrect == null) continue;

            Edge e = new Edge();
            e.setnode(stmt);
            stmt.add_edge(e);

            FactorNode fn = new FactorNode(stmt, e, pCorrect);
            e.setfactor(fn);

            pgraph.factornodes.add(fn);
            addedFactors.add(fn);
            added++;
        }

        System.out.println("[SPL][StmtPrior] added unary stmt priors = " + added);
        return addedFactors;
    }

    /**
     * 加载簇过滤文件：每行形如：
     *   <variantName> <rawTestName>
     * rawTestName 可能是 "ElevatorSystem::Elevator_ESTest.test00"，
     * 在这里会被规范化成 "ElevatorSystem::Elevator_test00"。
     */
    private static Map<String, Set<String>> loadClusterFilter(String filterPath) {
        Map<String, Set<String>> result = new HashMap<>();
        File f = new File(filterPath);
        if (!f.exists()) {
            System.err.println("[SPL][WARN] cluster filter file not found: " + filterPath);
            return result;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;

                String[] arr = line.split("\\s+");
                if (arr.length < 2) continue;

                String variantName = arr[0];
                String rawTestName = arr[1];  // 例如 "ElevatorSystem::Elevator_ESTest.test00"

                String normTestName = normalizeClusterTestName(rawTestName);

                result
                        .computeIfAbsent(variantName, k -> new HashSet<>())
                        .add(normTestName);

                System.out.println("[SPL][Cluster] add test: " + variantName
                        + "  " + rawTestName + "  ->  " + normTestName);
            }
        } catch (IOException e) {
            System.err.println("[SPL][ERROR] fail to read cluster filter: " + filterPath);
            e.printStackTrace();
        }
        return result;
    }

    /**
     * 从 mutant 根目录读取特征先验文件 feature_su_priors.txt
     * 格式：每行 "<FeatureName> <p_correct>"
     * 例如：Weight 0.73
     */
    private static Map<String, Double> loadFeaturePriorsFromFile(File mutantRoot) {
        Map<String, Double> priors = new HashMap<>();

        if (mutantRoot == null) {
            System.err.println("[SPL][Feature] mutantRoot is null, skip feature priors.");
            return priors;
        }

        File f = new File(mutantRoot, "feature_su_priors.txt");
        if (!f.exists()) {
            System.out.println("[SPL][Feature] feature priors file not found: " + f.getAbsolutePath());
            return priors;
        }

        System.out.println("[SPL][Feature] loading feature priors from: " + f.getAbsolutePath());

        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;

                String[] parts = line.split("\\s+");
                if (parts.length < 2) continue;

                String featName = parts[0];
                double p;
                try {
                    p = Double.parseDouble(parts[1]);
                } catch (NumberFormatException e) {
                    System.err.println("[SPL][Feature][WARN] bad prior line: " + line);
                    continue;
                }

                // 简单 clip 一下，保证在 [0,1] 内
                if (p < 0.0) p = 0.0;
                if (p > 1.0) p = 1.0;

                priors.put(featName, p);
            }
        } catch (IOException e) {
            System.err.println("[SPL][Feature][ERR] fail to read feature priors file: " + f.getAbsolutePath());
            e.printStackTrace();
        }

        System.out.println("[SPL][Feature] loaded feature priors: " + priors);
        return priors;
    }

    private static void debugFindStmt(ByteCodeGraph pgraph, String cls, int line) {
        int hit = 0;
        for (StmtNode s : pgraph.stmts) {
            String name = s.getName(); // e.g. BonusPoints.ExamDataBaseImpl:setBonusPoints:(II)V#14#13
            if (name != null && name.startsWith(cls + ":") && name.contains("#" + line + "#")) {
                if (hit < 10) System.out.println("[DEBUG][STMT] " + name);
                hit++;
            }
        }
        System.out.println("[DEBUG] stmts matched " + cls + " line " + line + " => " + hit);
    }

    /* ===================== 4. 主流程 ===================== */

    public static void main(String[] args) {
        // ====== 1) 解析参数：前面是 variantRoot，最后两个是 clusterId + filterFile ======
        if (args.length < 3) {
            System.err.println("Usage: SplGraphBuilder <variantRoot1> [<variantRoot2> ...] <clusterId> <clusterFilterFile>");
            System.exit(1);
        }

        // 最后两个参数：簇编号 + 过滤文件
        String clusterId = args[args.length - 2];
        String clusterFilterFile = args[args.length - 1];

        // 前面的都是 variantRoot
        int variantCount = args.length - 2;
        String[] variantRootArgs = new String[variantCount];
        System.arraycopy(args, 0, variantRootArgs, 0, variantCount);

        System.out.println("=== SPL Global GraphBuilder for " + variantCount + " variants, cluster=" + clusterId + " ===");
        for (String v : variantRootArgs) {
            System.out.println("  - variantRoot: " + v);
        }
        System.out.println("  - clusterFilterFile: " + clusterFilterFile);

        String globalName = deduceGlobalName(variantRootArgs[0]) + "_Cluster" + clusterId;
        System.out.println("[SPL] global name for result files: " + globalName);

        // ====== 2) 读取簇过滤文件：variantName -> 本簇中的 testName 集合 ======
        Map<String, Set<String>> clusterFilter = loadClusterFilter(clusterFilterFile);

        // 全局因子图
        ByteCodeGraph pgraph = new ByteCodeGraph();
        pgraph.setAutoOracle(true);
        pgraph.useD4jTest = true;
        pgraph.setResultFilter(false);
        //pgraph.setResultLogger("InfResult-"   + globalName);
//         pgraph.setGraphLogger("ProbGraph-"    + globalName);
//         pgraph.setReduceLogger("ReduceStmt-"  + globalName);

        // ====== 2.x) 预先把簇中的所有测试注册为 FAIL 测试 ======
        int preFailCount = 0;
        for (Map.Entry<String, Set<String>> e : clusterFilter.entrySet()) {
            for (String t : e.getValue()) {
                pgraph.d4jMethodNames.add(t);
                pgraph.d4jTriggerTestNames.add(t);  // 这些都是 FAIL
                preFailCount++;
            }
        }
        System.out.println("[SPL][Cluster] pre-registered failing tests from clusters = " + preFailCount);

        // ====== 3) 为每个 variantRoot 构造 Feature 映射器，并重写日志到 *_featured 目录 ======
        File[] variantRoots = new File[variantCount];
        File[] mytraceFeaturedDirs = new File[variantCount];
        File[] runFeaturedDirs = new File[variantCount];

        for (int i = 0; i < variantCount; i++) {
            File variantRoot = new File(variantRootArgs[i]);
            variantRoots[i] = variantRoot;

            System.out.println("[SPL] build feature mapper for variant: " + variantRoot);
            FeatureLocationMapper mapper = new FeatureLocationMapper(variantRoot);

            File mytraceFeatured = rewriteSourceLogsForVariant(variantRoot, mapper);
            File runFeatured     = rewriteRunLogsForVariant(variantRoot, mapper);

            mytraceFeaturedDirs[i] = mytraceFeatured;
            runFeaturedDirs[i]     = runFeatured;
        }

        // ---------- 4) 静态解析：遍历所有 *_featured 的 .source.log ----------

// 先收集“在动态日志中真正执行过”的类名集合（动态驱动静态图）
        Set<String> dynamicClasses = collectDynamicClasses(mytraceFeaturedDirs);

// 记录已经解析过的 "类键"，避免在不同变体里重复解析同一个类
        Set<String> parsedClassKeys = new HashSet<>();

        for (File mytraceDir : mytraceFeaturedDirs) {
            if (mytraceDir == null || !mytraceDir.exists() || !mytraceDir.isDirectory()) {
                System.err.println("[SPL][WARN] skip mytrace dir: " + mytraceDir);
                continue;
            }
            File[] fs = mytraceDir.listFiles();
            if (fs == null || fs.length == 0) {
                System.err.println("[SPL][WARN] no source logs in: " + mytraceDir);
                continue;
            }

            for (File f : fs) {
                String fname = f.getName();
                if (!fname.endsWith(".source.log") || fname.endsWith("traced.source.log")) {
                    continue;
                }

                // 去掉尾部 ".source.log"，得到类名部分
                // 例如：EmailSystem.Client.source.log -> "EmailSystem.Client"
                String baseName = fname.substring(0, fname.length() - ".source.log".length());

                // 1) 动态驱动过滤：如果某个类在动态日志中从未出现过，直接跳过，避免构建死代码的静态结构
                if (!dynamicClasses.isEmpty() && !dynamicClasses.contains(baseName)) {
                    // System.out.println("[SPL][SkipStaticNoDyn] " + baseName);
                    continue;
                }

                // 2) 过滤测试类（比 contains("test") 更稳一点）
                String lowerBase = baseName.toLowerCase();
                if (lowerBase.contains("junit") ||
                        lowerBase.contains("estest") ||     // 例如 EmailSystem.Client_ESTest
                        lowerBase.endsWith("test") ||
                        lowerBase.endsWith("tests")) {
                    // System.out.println("[SPL][SkipTestClass] " + baseName);
                    continue;
                }

                // 3) 再按 “类键” 去重：同一个类（在 feature 空间下）只解析一次
                String classKey = baseName;  // 如 "EmailSystem.Client"
                if (!parsedClassKeys.add(classKey)) {
                    // 说明这个类已经解析过了，只在第一个变体里用它作为静态结构即可
                    // System.out.println("[SPL][SkipDupClass] " + fname);
                    continue;
                }

                try {
                    System.out.println("[SPL] parsesource: " + f.getAbsolutePath());
                    pgraph.parsesource(f.getAbsolutePath());
                } catch (RuntimeException e) {
                    System.err.println("[SPL][WARN] skip malformed source log: " + fname);
                    e.printStackTrace();
                }
            }
        }

        System.out.println("[SPL] Parse source complete (all variants, featured)");

        // ---------- 5) 全局静态分析 ----------
        pgraph.get_pre_idom();
        pgraph.find_loop();
        pgraph.get_idom();
        pgraph.get_stores();
        System.out.println("[SPL] Static analyze complete (global)");

        // ---------- 6) 解析所有 variant 的动态 trace（只对本簇中有测试的变体） ----------
        for (int i = 0; i < variantCount; i++) {
            File mytraceDir = mytraceFeaturedDirs[i];
            File runDir     = runFeaturedDirs[i];

            if (runDir == null || !runDir.exists() || !runDir.isDirectory()) {
                System.err.println("[SPL][WARN] skip parseFolder, run_featured not found: " + runDir);
                continue;
            }

            String variantName = variantRoots[i].getName();
            Set<String> allowedTestsForVariant = clusterFilter.get(variantName);
            if (allowedTestsForVariant == null || allowedTestsForVariant.isEmpty()) {
                // 该变体在本簇中没有测试 ⇒ 不需要解析它的 run 日志
                System.out.println("[SPL] skip variant (no tests in this cluster): " + variantName);
                continue;
            }

            String runFolder  = runDir.getAbsolutePath();
            String sourceBase = (mytraceDir != null) ? mytraceDir.getAbsolutePath() : "";

            System.out.println("[SPL] parseFolder (cluster=" + clusterId + "): run=" + runFolder + ", sourceBase=" + sourceBase);
            // 这里 parseFolder 不再做额外过滤，是否 FAIL 完全由 d4jMethodNames / d4jTriggerTestNames 决定
            pgraph.parseFolder(runFolder, sourceBase, /*usesimple=*/false);
        }

        System.out.println("[SPL] Parse dynamic logs complete (cluster=" + clusterId + ", featured)");
        System.out.println("[SPL][DEBUG] graph stats: "
                + "stmts=" + pgraph.stmts.size()
                + ", vars=" + pgraph.nodes.size()
                + ", factors=" + pgraph.factornodes.size());

        System.out.println("[SPL][DEBUG] factorCount=" + pgraph.debugFactorCount
                + ", observedVars=" + pgraph.debugObservedVars
                + ", missingInsts=" + pgraph.debugMissingInsts);

//         debugFindStmt(pgraph, "BonusPoints.ExamDataBaseImpl", 13);
//         debugFindStmt(pgraph, "BonusPoints.ExamDataBaseImpl", 14);
        // ---------- 7) 构建特征节点与语句-特征因子 ----------
        addFeatureFactors(pgraph);

        // ---------- 7.1) 通过第一个 variantRoot 反推 mutant 根目录 ----------
        File mutantRoot = null;
        if (variantRoots.length > 0 && variantRoots[0] != null) {
            File vr0 = variantRoots[0];          // .../_MultipleBugs_.NOB_1.ID_2/variants/model_m_ca4_0002
            File variantsDir = vr0.getParentFile();  // .../_MultipleBugs_.NOB_1.ID_2/variants
            if (variantsDir != null) {
                mutantRoot = variantsDir.getParentFile(); // .../_MultipleBugs_.NOB_1.ID_2
            }
        }

        // ---------- 7.3) 语句级先验：从 stmt_sbfl_priors.txt 读取（多指标） ----------
        //Map<String, Map<String, Double>> stmtPriorsByMetric = new HashMap<>();
//        if (mutantRoot != null) {
//
//        } else {
//            System.err.println("[SPL][StmtPrior][WARN] cannot locate mutant root, skip stmt priors.");
//        }
        File stmtPriorFile = new File(mutantRoot, "stmt_sbfl_priors.txt");
        Map<String, Map<String, Double>> stmtPriorsByMetric = loadStmtSbflPriorsMulti(stmtPriorFile.getAbsolutePath());
// 如果没读到任何指标：就跑一次“无语句先验”的 BP（保持兼容）
        if (stmtPriorsByMetric == null || stmtPriorsByMetric.isEmpty()) {
            System.out.println("[SPL][StmtPrior] no metric priors found, run BP once without stmt priors.");
            pgraph.printgraph();
            pgraph.check_bp(true);
        } else {
            // 多指标循环：每轮仅替换“语句先验因子”，不重建图、不重解析日志
            List<FactorNode> lastAddedStmtPriorFactors = new ArrayList<>();

            for (String metric : stmtPriorsByMetric.keySet()) {
                Map<String, Double> priors = stmtPriorsByMetric.get(metric);
                if (priors == null || priors.isEmpty()) {
                    System.out.println("[SPL][StmtPrior] metric=" + metric + " has empty priors, skip.");
                    continue;
                }

                // 1) 清掉上一轮新增的一元语句先验因子（避免叠加）
                if (!lastAddedStmtPriorFactors.isEmpty()) {
                    pgraph.factornodes.removeAll(lastAddedStmtPriorFactors);
                    lastAddedStmtPriorFactors.clear();
                }

                // 2) 挂当前指标的语句先验
                System.out.println("[SPL][StmtPrior] attach metric=" + metric + ", priors=" + priors.size());
                lastAddedStmtPriorFactors = addStmtSbflPriors(pgraph, priors);

                // 3) 切换输出 logger，避免覆盖（每个指标一套结果文件）
                String name = globalName + "_" + metric;
                pgraph.setResultLogger("InfResult-"  + name);
//                 pgraph.setGraphLogger("ProbGraph-"   + name);
//                 pgraph.setReduceLogger("ReduceStmt-" + name);

                // 4) BP
                pgraph.printgraph();
                pgraph.check_bp(true);
                System.out.println("[SPL] BP finished for " + name);
            }
        }

        WriterUtils.cleanup().run();
    }
}
