import java.io.File;
import java.lang.reflect.Method;
import java.lang.reflect.Field;
import java.util.*;
import slice.*;
import com.ibm.wala.classLoader.SourceDirectoryTreeModule;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.slicer.*;
import com.ibm.wala.ipa.modref.ModRef;
import com.ibm.wala.ssa.*;

/** 调用官方语句切片和字段传播函数，复用只读邻接并导出静态关系。 */
public class VarcopFeatureProfile {
    static long started=System.nanoTime();
    static void log(String phase) {System.out.printf(Locale.ROOT,"%.3f %s%n",(System.nanoTime()-started)/1e9,phase);}
    /** 图完全构建后只读复用邻接枚举；结束时逐项重新读取原图，拒绝任何变化。 */
    static class CachedGraph extends SDG<InstanceKey> {
        boolean enabled=false;
        long hits=0;
        final HashMap<Statement,List<Statement>> next=new HashMap<>(),previous=new HashMap<>();
        @SuppressWarnings("unchecked") CachedGraph(CallGraph graph,PointerAnalysis<?> analysis) {
            super(graph,(PointerAnalysis<InstanceKey>)analysis,ModRef.make(),Slicer.DataDependenceOptions.NO_HEAP,
                  Slicer.ControlDependenceOptions.NO_EXCEPTIONAL_EDGES,null);
        }
        static List<Statement> collect(Iterator<Statement> iterator) {
            List<Statement> result=new ArrayList<>();iterator.forEachRemaining(result::add);return result;
        }
        @Override public Iterator<Statement> getSuccNodes(Statement statement) {
            if(!enabled)return super.getSuccNodes(statement);
            List<Statement> values=next.get(statement);
            if(values==null) {values=collect(super.getSuccNodes(statement));next.put(statement,values);} else hits++;
            return values.iterator();
        }
        @Override public Iterator<Statement> getPredNodes(Statement statement) {
            if(!enabled)return super.getPredNodes(statement);
            List<Statement> values=previous.get(statement);
            if(values==null) {values=collect(super.getPredNodes(statement));previous.put(statement,values);} else hits++;
            return values.iterator();
        }
        void verifyUnchanged() {
            for(Map.Entry<Statement,List<Statement>> entry:next.entrySet())
                if(!entry.getValue().equals(collect(super.getSuccNodes(entry.getKey()))))
                    throw new IllegalStateException("Successor adjacency changed during slicing");
            for(Map.Entry<Statement,List<Statement>> entry:previous.entrySet())
                if(!entry.getValue().equals(collect(super.getPredNodes(entry.getKey()))))
                    throw new IllegalStateException("Predecessor adjacency changed during slicing");
            log("adjacency verified; hits="+hits+" cached="+(next.size()+previous.size()));
        }
    }
    static String encode(String value) {
        return Base64.getEncoder().encodeToString(value.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }
    static void exportBasis(String filename) throws Exception {
        HashMap<String,HashSet<String>> features=StatementContainer.featureForwardInfluenceContainer();
        HashMap<String,Statement> roots=StatementContainer.getLineKeyToForwardStatementMapping();
        HashSet<String> keys=new HashSet<>();
        java.io.PrintWriter out=new java.io.PrintWriter(new java.io.OutputStreamWriter(
            new java.util.zip.GZIPOutputStream(new java.io.FileOutputStream(filename)),
            java.nio.charset.StandardCharsets.UTF_8));
        for(Map.Entry<String,HashSet<String>> feature:features.entrySet()) {
            if(feature.getValue().isEmpty())out.println("Z\t"+encode(feature.getKey()));
            for(String key:feature.getValue()) {
                out.println("F\t"+encode(feature.getKey())+"\t"+encode(key));
                keys.add(key);
            }
        }
        HashSet<String> mappingKeys=new HashSet<>(keys);
        for(String key:keys) {
            Statement statement=roots.get(key);
            HashSet<Statement> dependencies=StatementContainer.obtainBackwardSlicingStatements(statement);
            if(dependencies==null) {
                out.println("N\t"+encode(key));
                continue;
            }
            for(Statement dependency:dependencies) {
                String dep=dependency.getLineKey();
                out.println("B\t"+encode(key)+"\t"+encode(dep));
                mappingKeys.add(dep);
            }
        }
        for(String key:mappingKeys) {
            SPLFeatureSourceLine value=MyECJJavaSourceAnalysisEngine.getFeatureLineInfo(key);
            if(value!=null)out.println("M\t"+encode(key)+"\t"+encode(value.toString()));
        }
        for(String value:StatementContainer.executedFeatureLineSet)out.println("E\t"+encode(value));
        out.close();
    }
    @SuppressWarnings({"rawtypes","unchecked"})
    public static void main(String[] args) throws Exception {
        String source=args[0];
        System.setProperty("spc_path",args[1]);
        FileUtil.readUniqueSPCsFromFile();
        SliceFromStatement.init(source);
        System.setProperty("wala.jdt.quiet","true");
        MyECJJavaSourceAnalysisEngine engine=new MyECJJavaSourceAnalysisEngine();
        engine.addSystemDependencies();
        engine.addSourceModule(new SourceDirectoryTreeModule(new File(source)));
        engine.setExclusionsFile("Java60RegressionExclusions.txt");
        engine.buildAnalysisScope(); log("scope");
        CallGraph graph=engine.buildDefaultCallGraph(); log("call graph");
        CachedGraph sdg=new CachedGraph(graph,engine.getPointerAnalysis());
        sdg.eagerConstruction();log("dependence graph");
        sdg.enabled=Boolean.parseBoolean(System.getProperty("cached.adjacency","true"));
        Method seeds=SliceFromStatement.class.getDeclaredMethod("getAllStatements",SDG.class);
        seeds.setAccessible(true);
        List<Statement> statements=(List<Statement>)seeds.invoke(null,sdg);
        // 官方addForward/BackwardSlicingStatements在根未执行时只初始化空集合，丢弃传入的切片。
        // 因而保留全部根及其顺序，只跳过必然被丢弃的查询；依赖图已完整构建且仍包含全部语句。
        boolean skip=Boolean.parseBoolean(System.getProperty("skip.unexecuted","true"));
        List<Statement> puts=new ArrayList<>();
        int omitted=0;
        for(Statement statement:statements) {
            boolean executed=StatementContainer.isStatementExecuted(statement);
            boolean omit=skip&&!executed;
            StatementContainer.addForwardSlicingStatements(statement,
                omit?Collections.emptyList():Slicer.computeForwardSlice(sdg,statement));
            StatementContainer.addBackwardSlicingStatements(statement,
                omit?Collections.emptyList():Slicer.computeBackwardSlice(sdg,statement));
            if(omit)omitted++;
            SSAInstruction instruction=PDG.statement2SSAInstruction(statement.getNode().getIR().getInstructions(),statement);
            if(instruction instanceof SSAPutInstruction && executed)puts.add(statement);
        }
        SliceFromStatement.slicePutAttributeStatements(puts);
        log("seed slicing and official field propagation finished; skipped="+omitted+"/"+statements.size());
        if(sdg.enabled)sdg.verifyUnchanged();
        if(System.getProperty("export.basis")!=null)exportBasis(System.getProperty("export.basis"));
        java.nio.file.Path groups=java.nio.file.Paths.get(args[2]);
        for(String record:java.nio.file.Files.readAllLines(groups,java.nio.charset.StandardCharsets.UTF_8)) {
            String[] pair=record.split("\\t",2);
            FileUtil.ByFileSPCs.clear();
            System.setProperty("spc_path",pair[0]);
            FileUtil.readUniqueSPCsFromFile();
            if(!FileUtil.ByFileSPCs.containsKey(source))continue;
            StatementContainer.interactionLineCounter.clear();
            Field output=SliceFromStatement.class.getDeclaredField("finalOutput");
            output.setAccessible(true);((List)output.get(null)).clear();
            Method calculate=SliceFromStatement.class.getDeclaredMethod("calculateEachSourceFeatureInteractions",String.class);
            calculate.setAccessible(true);calculate.invoke(null,source);
            Method save=SliceFromStatement.class.getDeclaredMethod("saveFeatureInteractionsOutput",String.class);
            save.setAccessible(true);save.invoke(null,source);
            System.setProperty("slicing_output_path",pair[1]);
            Method print=SliceFromStatement.class.getDeclaredMethod("printFinalFeatureInteractions");
            print.setAccessible(true);print.invoke(null);
        }
        log("saved groups");
        SliceFromStatement.reset();
    }
}
