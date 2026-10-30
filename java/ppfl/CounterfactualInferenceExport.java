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
        if (args.length != 5) {
            throw new IllegalArgumentException("sourceDir traceFile testName PASS|FAIL output");
        }
        File sourceDir = new File(args[0]);
        File traceFile = new File(args[1]);
        String testName = args[2];
        boolean passed = "PASS".equalsIgnoreCase(args[3]);
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
        // 真正的概率推理入口：使用移植BRFL的循环信念传播实现。
        graph.bp_inference();

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
                    + node.bp_getprob() + "\t" + (bool(node, "obs") ? "1" : "0") + "\n");
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
        WriterUtils.cleanup().run();
    }
}
