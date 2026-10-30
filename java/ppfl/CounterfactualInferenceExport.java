package ppfl;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;
import java.nio.file.Files;
import java.nio.file.Paths;

/** 构建并推断BRFL原生因子图，然后无损导出节点、因子、参数表和关联角色。 */
public final class CounterfactualInferenceExport {
    private static Field field(Class<?> type, String name) throws Exception {
        Field value = type.getDeclaredField(name);
        value.setAccessible(true);
        return value;
    }

    @SuppressWarnings("unchecked")
    private static <T> T get(Object object, String name) throws Exception {
        return (T) field(object.getClass(), name).get(object);
    }

    private static String encode(String text) {
        if (text == null) text = "";
        return Base64.getUrlEncoder().withoutPadding().encodeToString(text.getBytes(StandardCharsets.UTF_8));
    }

    private static String join(List<?> values, String separator) {
        if (values == null) return "";
        StringBuilder out = new StringBuilder();
        for (Object value : values) {
            if (out.length() > 0) out.append(separator);
            out.append(String.valueOf(value));
        }
        return out.toString();
    }

    private static boolean bool(Node node, String field) throws Exception {
        return ((Boolean) CounterfactualInferenceExport.field(Node.class, field).get(node)).booleanValue();
    }

    private static String id(Node node, IdentityHashMap<Node, String> ids) {
        return ids.get(node);
    }

    private static void incidence(BufferedWriter out, String factor, Node node, String role,
                                  IdentityHashMap<Node, String> ids) throws Exception {
        if (node != null && id(node, ids) != null) {
            out.write("INCIDENCE\t" + factor + "\t" + id(node, ids) + "\t" + role + "\n");
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 3 && args[0].equals("--joint")) {
            joint(args[1], new File(args[2]));
            return;
        }
        if (args.length != 5 && args.length != 6) {
            throw new IllegalArgumentException("sourceDir traceFile testName PASS|FAIL output");
        }
        File sourceDir = new File(args[0]);
        File traceFile = new File(args[1]);
        String testName = args[2];
        String outcome = args[3].toUpperCase(java.util.Locale.ROOT);
        if (!outcome.equals("PASS") && !outcome.equals("PASSED")
                && !outcome.equals("FAIL") && !outcome.equals("FAILED")) {
            throw new IllegalArgumentException("unknown test outcome: " + args[3]);
        }
        boolean passed = outcome.equals("PASS") || outcome.equals("PASSED");
        File output = new File(args[4]);

        // 每次调用只接收一个测试执行轨迹；当前不合并A/B或多测试证据。
        ByteCodeGraph graph = new ByteCodeGraph();
        graph.setAutoOracle(true);
        graph.setTraceAllClassed(true);
        File[] sources = sourceDir.listFiles();
        if (sources == null) throw new IllegalArgumentException("source logs missing: " + sourceDir);
        for (File source : sources) {
            if (source.getName().endsWith(".source.log") && !source.getName().equals("traced.source.log")) {
                graph.parsesource(source.getAbsolutePath());
            }
        }
        graph.get_pre_idom();
        graph.find_loop();
        graph.get_idom();
        graph.get_stores();
        graph.parsetrace(traceFile.getAbsolutePath(), testName, passed);
        // 统一模式下这里只解析轨迹，不计算单侧后验。
        if (args.length == 5) graph.bp_inference();
        export(graph, output);
        WriterUtils.cleanup().run();
    }

    private static void export(ByteCodeGraph graph, File output) throws Exception {

        IdentityHashMap<Node, String> ids = new IdentityHashMap<Node, String>();
        int index = 0;
        for (StmtNode stmt : graph.stmts) ids.put(stmt, "s" + index++);
        index = 0;
        for (Node node : graph.nodes) ids.put(node, "v" + index++);

        output.getParentFile().mkdirs();
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output), StandardCharsets.UTF_8));
        out.write("#BRFL_FACTOR_GRAPH\t2\n");
        for (StmtNode stmt : graph.stmts) {
            out.write("STATEMENT\t" + id(stmt, ids) + "\t" + encode(stmt.getName()) + "\t"
                    + stmt.bp_getprob() + "\t" + (bool(stmt, "obs") ? "1" : "0") + "\t"
                    + (stmt.getreduced() ? "1" : "0") + "\n");
        }
        for (Node node : graph.nodes) {
            String[] stack = node.getStackValue();
            String stackText = stack == null ? "" : join(java.util.Arrays.asList(stack), "|");
            Node origin = (Node) field(Node.class, "stmt").get(node);
            out.write("VALUE\t" + id(node, ids) + "\t" + encode(node.getName()) + "\t"
                    + encode(origin == null ? "" : origin.getName()) + "\t" + encode(stackText) + "\t"
                    + node.bp_getprob() + "\t" + (bool(node, "obs") ? "1" : "0") + "\t"
                    + (bool(node, "obsvalue") ? "1" : "0") + "\n");
        }
        index = 0;
        for (FactorNode factor : graph.factornodes) {
            String factorId = "f" + index++;
            Node stmt = get(factor, "stmt");
            Node def = get(factor, "def");
            List<Node> preds = get(factor, "preds");
            List<Node> uses = get(factor, "uses");
            List<String> ops = get(factor, "ops");
            List<Double> tensor = get(factor, "tensor");
            out.write("FACTOR\t" + factorId + "\t" + encode(stmt == null ? "" : stmt.getName()) + "\t"
                    + encode(join(ops, "|")) + "\t" + encode(join(tensor, ",")) + "\n");
            incidence(out, factorId, stmt, "STMT", ids);
            incidence(out, factorId, def, "DEF", ids);
            if (preds != null) for (Node node : preds) incidence(out, factorId, node, "PRED", ids);
            if (uses != null) for (Node node : uses) incidence(out, factorId, node, "USE", ids);
        }
        out.close();
    }

    /** 独立解析各执行，统一静态语句身份后只推理一次。manifest每行：
     * 执行标识、静态日志目录、轨迹、测试名、PASS/FAIL、源码映射文件。
     */
    @SuppressWarnings("unchecked")
    private static void joint(String manifest, File output) throws Exception {
        ByteCodeGraph merged = new ByteCodeGraph();
        Map<String, StmtNode> shared = new LinkedHashMap<String, StmtNode>();
        for (String row : Files.readAllLines(Paths.get(manifest), StandardCharsets.UTF_8)) {
            String[] a = row.split("\t", -1);
            if (a.length != 6) throw new IllegalArgumentException("invalid execution manifest");
            Map<String, String> mapping = new LinkedHashMap<String, String>();
            for (String line : Files.readAllLines(Paths.get(a[5]), StandardCharsets.UTF_8)) {
                String[] pair = line.split("\t", -1);
                mapping.put(pair[0], pair[1]);
            }
            ppfl.instrumentation.RuntimeFrame.resetFrames();
            ByteCodeGraph local = new ByteCodeGraph();
            local.setAutoOracle(true);
            local.setTraceAllClassed(true);
            File[] sources = new File(a[1]).listFiles();
            if (sources == null) throw new IllegalArgumentException("missing source logs");
            for (File source : sources) {
                if (source.getName().endsWith(".source.log") && !source.getName().equals("traced.source.log"))
                    local.parsesource(source.getAbsolutePath());
            }
            local.get_pre_idom(); local.find_loop(); local.get_idom(); local.get_stores();
            local.parsetrace(a[2], a[3], a[4].equals("PASS"));
            IdentityHashMap<Node, StmtNode> replacements = new IdentityHashMap<Node, StmtNode>();
            for (StmtNode stmt : local.stmts) {
                String raw = stmt.getName();
                String[] parts = raw.split("#");
                String runtimeClass = parts[0].split(":")[0].replace('/', '.');
                String identity = parts.length > 1 ? mapping.get(runtimeClass + "#" + parts[1]) : null;
                // 缺少特征映射的节点不跨执行合并，避免把测试或不同特征实现误合并。
                if (identity == null) identity = "unmapped:" + a[0] + ":" + raw;
                StmtNode canonical = shared.get(identity);
                if (canonical == null) {
                    canonical = new StmtNode(identity, stmt.form);
                    shared.put(identity, canonical);
                    merged.stmts.add(canonical);
                }
                // 测试代码被BRFL视为正确的已知语句；合并时保留其观测，不能丢失。
                if (bool(stmt, "obs")) {
                    if (bool(canonical, "obs") && canonical.obsvalue != stmt.obsvalue)
                        throw new IllegalStateException("conflicting statement observation: " + identity);
                    canonical.observe(stmt.obsvalue);
                }
                replacements.put(stmt, canonical);
            }
            for (Node value : local.nodes) {
                value.name = a[0] + ":" + value.name;
                if (value.stmt != null) value.stmt = replacements.get(value.stmt);
                merged.nodes.add(value);
            }
            for (FactorNode factor : local.factornodes) {
                Node old = get(factor, "stmt");
                StmtNode canonical = replacements.get(old);
                if (canonical == null) throw new IllegalStateException("factor has unmapped statement");
                field(FactorNode.class, "stmt").set(factor, canonical);
                List<Node> all = get(factor, "allnodes");
                if (all != null) for (int i = 0; i < all.size(); i++)
                    if (replacements.containsKey(all.get(i))) all.set(i, replacements.get(all.get(i)));
                List<Edge> edges = get(factor, "alledges");
                for (Edge edge : edges) {
                    StmtNode replacement = replacements.get(edge.getnode());
                    if (replacement != null) { edge.setnode(replacement); replacement.add_edge(edge); }
                }
                merged.factornodes.add(factor);
            }
        }
        if (merged.nodes.isEmpty() || merged.stmts.isEmpty()) throw new IllegalStateException("empty joint graph");
        merged.bp_inference();
        export(merged, output);
        WriterUtils.cleanup().run();
    }
}
