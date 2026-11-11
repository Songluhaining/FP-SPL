import java.io.File;
import java.lang.reflect.Method;
import java.lang.reflect.Field;
import java.util.*;
import slice.*;
import com.ibm.wala.classLoader.SourceDirectoryTreeModule;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.slicer.*;
import com.ibm.wala.ipa.modref.ModRef;
import com.ibm.wala.ssa.*;

/** 调用原切片实现，记录各阶段耗时；可跳过最终必然丢弃的未执行根语句计算。 */
public class SliceProfile {
    static long started=System.nanoTime();
    static void log(String phase) {System.out.printf(Locale.ROOT,"%.3f %s%n",(System.nanoTime()-started)/1e9,phase);}
    /** 原实现每次加边扫描所有根；这里维护“某语句出现在哪些根的切片里”的精确索引。 */
    static class Indexed {
        final HashMap<Statement,HashSet<Statement>> map;
        final HashMap<Statement,Integer> numbers=new HashMap<>();
        final ArrayList<Statement> roots=new ArrayList<>();
        final HashMap<Statement,BitSet> reverse=new HashMap<>();
        final HashMap<Statement,Boolean> executed=new HashMap<>();
        @SuppressWarnings("unchecked") Indexed(String field) throws Exception {
            Field f=StatementContainer.class.getDeclaredField(field);f.setAccessible(true);
            map=(HashMap<Statement,HashSet<Statement>>)f.get(null);
            for(Statement root:map.keySet()) {
                int id=number(root);
                for(Statement item:map.get(root))reverse.computeIfAbsent(item,k->new BitSet()).set(id);
            }
        }
        int number(Statement root) {
            Integer id=numbers.get(root);
            if(id==null) {id=roots.size();roots.add(root);numbers.put(root,id);}
            return id;
        }
        void add(Statement root,Collection<Statement> items,boolean filter) {
            HashSet<Statement> values=map.computeIfAbsent(root,k->new HashSet<>());
            int id=number(root);
            for(Statement item:items) {
                if(filter && !executed.computeIfAbsent(item,StatementContainer::isStatementExecuted))continue;
                if(values.add(item))reverse.computeIfAbsent(item,k->new BitSet()).set(id);
            }
        }
        void propagate(Statement root,Collection<Statement> items) {
            map.computeIfAbsent(root,k->new HashSet<>());
            number(root);
            if(executed.computeIfAbsent(root,StatementContainer::isStatementExecuted))add(root,items,true);
            BitSet related=reverse.get(root);
            if(related!=null) {
                BitSet snapshot=(BitSet)related.clone();
                for(int id=snapshot.nextSetBit(0);id>=0;id=snapshot.nextSetBit(id+1))add(roots.get(id),items,false);
            }
        }
    }
    static void propagate(List<Statement> puts) throws Exception {
        Indexed forward=new Indexed("forwardSliceMap");
        Indexed backward=new Indexed("backwardSliceMap");
        HashMap<Statement,HashSet<Statement>> propagated=new HashMap<>();
        for(Statement root:puts) {
            HashSet<Statement> added=SliceFromStatement.slicePutStatement(root,new ArrayList<>());
            forward.propagate(root,added);propagated.put(root,added);
        }
        for(Statement root:propagated.keySet()) {
            Set<Statement> singleton=Collections.singleton(root);
            for(Statement item:propagated.get(root))backward.propagate(item,singleton);
        }
    }
    @SuppressWarnings({"rawtypes","unchecked"})
    public static void main(String[] args) throws Exception {
        String source=args[0];
        SliceFromStatement.init(source);
        System.setProperty("wala.jdt.quiet","true");
        MyECJJavaSourceAnalysisEngine engine=new MyECJJavaSourceAnalysisEngine();
        engine.addSystemDependencies();
        engine.addSourceModule(new SourceDirectoryTreeModule(new File(source)));
        engine.setExclusionsFile("Java60RegressionExclusions.txt");
        engine.buildAnalysisScope(); log("scope");
        CallGraph graph=engine.buildDefaultCallGraph(); log("call graph");
        SDG sdg=new SDG(graph,engine.getPointerAnalysis(),ModRef.make(),Slicer.DataDependenceOptions.NO_HEAP,
                        Slicer.ControlDependenceOptions.NO_EXCEPTIONAL_EDGES,null);
        sdg.eagerConstruction();log("dependence graph");
        Method seeds=SliceFromStatement.class.getDeclaredMethod("getAllStatements",SDG.class);
        seeds.setAccessible(true);
        List<Statement> statements=(List<Statement>)seeds.invoke(null,sdg);
        List<Statement> puts=new ArrayList<>();
        boolean skip=Boolean.getBoolean("skip.unexecuted");
        int index=0;
        for (Statement s:statements) {
            boolean executed=StatementContainer.isStatementExecuted(s);
            StatementContainer.addForwardSlicingStatements(s,skip&&!executed?Collections.emptyList():Slicer.computeForwardSlice(sdg,s));
            StatementContainer.addBackwardSlicingStatements(s,skip&&!executed?Collections.emptyList():Slicer.computeBackwardSlice(sdg,s));
            SSAInstruction instruction=PDG.statement2SSAInstruction(s.getNode().getIR().getInstructions(),s);
            if(instruction instanceof SSAPutInstruction && executed)puts.add(s);
            if(++index%100==0)log("seeds "+index+"/"+statements.size());
        }
        log("seed slicing finished; puts="+puts.size());
        if(Boolean.getBoolean("indexed.propagation"))propagate(puts);
        else SliceFromStatement.slicePutAttributeStatements(puts);
        log("field propagation");
        Method save=SliceFromStatement.class.getDeclaredMethod("saveFinalOutput",String.class);
        save.setAccessible(true);save.invoke(null,source);log("saved");
        SliceFromStatement.reset();
    }
}
