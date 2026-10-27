package ppfl;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.List;

/** Export BRFL's per-test runtime factors before its fault-localization inference. */
public final class FpsplTraceExport {
    private static final Field PREDS;
    private static final Field USES;
    private static final Field DEF;

    static {
        try {
            PREDS = FactorNode.class.getDeclaredField("preds");
            USES = FactorNode.class.getDeclaredField("uses");
            DEF = FactorNode.class.getDeclaredField("def");
            PREDS.setAccessible(true);
            USES.setAccessible(true);
            DEF.setAccessible(true);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private FpsplTraceExport() { }

    @SuppressWarnings("unchecked")
    private static List<Node> nodes(Field field, FactorNode factor) throws IllegalAccessException {
        Object result = field.get(factor);
        return result == null ? new ArrayList<Node>() : (List<Node>) result;
    }

    private static String origin(Node node) {
        return node == null || node.stmt == null ? "" : node.stmt.getName();
    }

    private static String encode(String value) {
        return Base64.getEncoder().encodeToString(value.getBytes(StandardCharsets.UTF_8));
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            System.err.println("Usage: FpsplTraceExport <mytrace-dir> <run-dir> <test-oracle-name> <PASS|FAIL> <output-tsv>");
            System.exit(2);
        }
        File sourceDir = new File(args[0]);
        File runDir = new File(args[1]);
        if (!sourceDir.isDirectory() || !runDir.isDirectory()) {
            throw new IllegalArgumentException("BRFL source/run trace directories are missing");
        }
        ByteCodeGraph graph = new ByteCodeGraph();
        graph.useD4jTest = true;
        graph.setAutoOracle(true);
        graph.setResultFilter(false);
        graph.d4jMethodNames.add(args[2]);
        if ("FAIL".equals(args[3])) {
            graph.d4jTriggerTestNames.add(args[2]);
        } else if (!"PASS".equals(args[3])) {
            throw new IllegalArgumentException("Expected PASS or FAIL");
        }
        File[] sourceLogs = sourceDir.listFiles((dir, name) -> name.endsWith(".source.log")
                && !name.equals("traced.source.log"));
        if (sourceLogs == null || sourceLogs.length == 0) {
            throw new IllegalStateException("BRFL source logs are empty");
        }
        List<File> sorted = new ArrayList<>();
        for (File file : sourceLogs) sorted.add(file);
        sorted.sort(Comparator.comparing(File::getName));
        for (File file : sorted) graph.parsesource(file.getAbsolutePath());
        graph.get_pre_idom();
        graph.find_loop();
        graph.get_idom();
        graph.get_stores();
        graph.parseFolder(runDir.getAbsolutePath(), sourceDir.getAbsolutePath(), false);

        int factors = 0;
        int edges = 0;
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(args[4]))) {
            writer.write("META\tBRFL_RUNTIME_FACTOR_V1\t" + args[2] + "\t" + args[3]);
            writer.newLine();
            for (FactorNode factor : graph.factornodes) {
                Node stmt = factor.getstmt();
                if (!(stmt instanceof StmtNode) || DEF.get(factor) == null) continue;
                String target = stmt.getName();
                writer.write("NODE\t" + factors + "\t" + target);
                writer.newLine();
                Node defined = (Node) DEF.get(factor);
                if (defined.getStackValue() != null) {
                    writer.write("VALUE\t" + factors + "\t" + target + "\t"
                            + encode(String.join("|", defined.getStackValue())));
                    writer.newLine();
                }
                for (Node source : nodes(PREDS, factor)) {
                    String from = origin(source);
                    if (!from.isEmpty() && !from.equals(target)) {
                        writer.write("EDGE\t" + factors + "\t" + from + "\t" + target + "\tCONTROL");
                        writer.newLine();
                        edges++;
                    }
                }
                for (Node source : nodes(USES, factor)) {
                    String from = origin(source);
                    if (!from.isEmpty() && !from.equals(target)) {
                        writer.write("EDGE\t" + factors + "\t" + from + "\t" + target + "\tDATA");
                        writer.newLine();
                        edges++;
                    }
                }
                factors++;
            }
            writer.write("SUMMARY\t" + factors + "\t" + edges + "\t" + graph.debugMissingInsts);
            writer.newLine();
        }
        System.out.println("FP-SPL exported runtime factors=" + factors + " edges=" + edges);
        if (factors == 0) System.exit(3);
    }
}
