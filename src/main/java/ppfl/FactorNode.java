package ppfl;

import java.util.ArrayList;
import java.util.List;
import java.io.*;

// import org.apfloat.Apfloat;
// import org.apfloat.ApfloatMath;

// import org.slf4j.Logger;
// import org.slf4j.LoggerFactory;

public class FactorNode {

	protected static MyWriter debugLogger = WriterUtils.getWriter("Debugger");
	protected static MyWriter printLogger = WriterUtils.getWriter("GraphLogger");

	private List<Node> preds;
	private Node def;
	private Node stmt;
	private List<Node> uses;
	private List<String> ops;// TODO consider operators
	private static final String[] unkops = { "%", "<", "<=", ">", ">=", "==", "!=" };
	private double HIGH = 1.0; // can change to 1-1e-10
	private double VHIGH = 0.99999;
	private double MEDIUM_HIGH = 0.5;
	private double MEDIUM = 0.5;
	private double MEDIUM_LOW=0.5;
	private double LOW = 0.0; // can change to 1e-10
	private List<Double> tensor;
	// private Apfloat ap_HIGH = new Apfloat("1.0", 100);
	// private Apfloat ap_MEDIUM = new Apfloat("0.5", 100);
	// private Apfloat ap_LOW = new Apfloat("0.0", 100);
	// private List<Apfloat> ap_tensor;
	private boolean use_ap = false;
	private List<Node> allnodes;
	private List<Edge> alledges;
	private int nnodes;
	private double stmtvalue;
	boolean hasUNKoperator = true;

	private Edge dedge;
	private Edge sedge;
	private List<Edge> pedges;
	private List<Edge> uedges;
    // 语句–特征因子的默认参数：一致时高概率，不一致时低概率
    private static final double FEATURE_STMT_HIGH = 0.9;
    private static final double FEATURE_STMT_LOW  = 0.1;
	//private static List<Double> numbersArray2 = {0,2.3283064365e-10,5.4210108624e-20,0.00390625,1.5258789063e-5,0.3333333333,0.5};
	// private static List<Double> numbersArray2;
	private static List<Double> numbersArray;
	static {

		try {
			BufferedReader readTxt=new BufferedReader(new FileReader("./infer.txt"));
			String str = readTxt.readLine();
			String[] numbersArray_s=str.split(",");
			numbersArray = new ArrayList<>();
			for(String tmp : numbersArray_s)
				numbersArray.add(Double.parseDouble(tmp));

		} catch (IOException e) {
			e.printStackTrace();
		}
    }
	private static final int[][] spopcodes = {{108,109,110,111},{112,113,114,115},{126,127,128,129,130,131},{136,139,140,142,143,144,147},{148,149,150,151,152},{153,154,155,156,157,158},{159,160,161,162,163,164},{165,166},{198,199}}; 
	//private static final int[][] spopcodes ={{96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,137,139,142,144},{97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,129,131,135,138,140,141,143},{145},{146,147},{148,149,150,151,152},{153,154,155,156,157,158,159,160,161,162,163,164,165,166,198,199}};

	public FactorNode() {
		this.stmt = null;
		this.def = null;
		this.dedge = null;
	}

	// factor with only a stmt node
	public FactorNode(Node stmt, Edge sedge, double value) {
		this.stmt = stmt;
		this.def = null;
		this.dedge = null;
		this.sedge = sedge;
		this.tensor = new ArrayList<>();
		// this.ap_tensor = new ArrayList<>();
		this.nnodes = 1;
		this.stmtvalue = value;
		this.alledges = new ArrayList<>();
		alledges.add(sedge);
		if(use_ap){
			// ap_tensor.add(new Apfloat("1.0", 100).subtract(new Apfloat(String.valueOf(value),100)));
			// ap_tensor.add(new Apfloat(String.valueOf(value),100));
		}
		else{
			tensor.add(1 - value);
			tensor.add(value);
		}
	}

	public FactorNode(Node def, Node stmt, List<Node> preds, List<Node> uses, List<String> ops, Edge dedge, Edge sedge,
			List<Edge> pedges, List<Edge> uedges) {
		this.preds = preds;
		this.stmt = stmt;
		this.def = def;
		this.uses = uses;
		this.ops = ops;
		this.dedge = dedge;
		this.sedge = sedge;
		this.pedges = pedges;
		this.uedges = uedges;
		this.tensor = new ArrayList<>();
		// this.ap_tensor = new ArrayList<>();
		this.allnodes = new ArrayList<>();
		allnodes.add(stmt);
		allnodes.add(def);
		allnodes.addAll(preds);
		allnodes.addAll(uses);
		this.alledges = new ArrayList<>();
		alledges.add(sedge);
		alledges.add(dedge);
		alledges.addAll(pedges);
		alledges.addAll(uedges);
		this.nnodes = allnodes.size();
		if (this.ops != null)
			for (String op : this.ops) {
				for (String unk : unkops) {
					if (op.contentEquals(unk))
					this.hasUNKoperator = true;
				}
			}
		change_parameters();
		
		// if (use_ap)
		// 	ap_gettensor(allnodes, nnodes - 1);
		// else
			gettensor(allnodes, nnodes - 1);
	}

    /**
     * 新增：语句节点与特征节点之间的二元因子。
     *
     * 节点顺序约定：
     *   allnodes[0] = stmt
     *   allnodes[1] = feature
     *
     * tensor 索引的二进制展开（与 send_message 中一致）：
     *   index = 0 -> (stmt=false, feature=false)
     *   index = 1 -> (stmt=true , feature=false)
     *   index = 2 -> (stmt=false, feature=true)
     *   index = 3 -> (stmt=true , feature=true)
     *
     * 我们令：
     *   s,f 一致时权重 FEATURE_STMT_HIGH；
     *   s,f 不一致时权重 FEATURE_STMT_LOW。
     */
    public FactorNode(StmtNode stmt, Node feature,
                      Edge sedge, Edge fedge) {
        this.stmt = stmt;
        this.def = null;
        this.preds = null;
        this.uses  = null;
        this.ops   = null;

        this.dedge  = null;
        this.sedge  = sedge;
        this.pedges = null;
        this.uedges = null;

        this.allnodes = new ArrayList<>();
        this.allnodes.add(stmt);    // bit0
        this.allnodes.add(feature); // bit1

        this.alledges = new ArrayList<>();
        this.alledges.add(sedge);
        this.alledges.add(fedge);

        this.nnodes = allnodes.size();

        this.tensor = new ArrayList<>();

        // (stmt=false, feature=false)
        this.tensor.add(FEATURE_STMT_HIGH);

        // (stmt=true , feature=false)
        this.tensor.add(FEATURE_STMT_LOW);

        // (stmt=false, feature=true)
        this.tensor.add(FEATURE_STMT_LOW);

        // (stmt=true , feature=true)
        this.tensor.add(FEATURE_STMT_HIGH);
    }

    // 新增：可疑特征组合因子（只连若干 Feature / Node）
// 语义：
// - allCorrectWeight：组合中所有特征都为 true(正确) 的权重（要设得比较小，比如 0.2）
// - notAllCorrectWeight：其余任何状态（至少一个为 false）权重（设得大，比如 0.8）
    public FactorNode(List<Node> featureNodes,
                      List<Edge> featureEdges,
                      double allCorrectWeight,
                      double notAllCorrectWeight) {
        this.preds = null;
        this.def   = null;
        this.stmt  = null;
        this.uses  = null;
        this.ops   = null;
        this.dedge = null;
        this.sedge = null;
        this.pedges = null;
        this.uedges = null;

        this.allnodes = new ArrayList<>(featureNodes);
        this.alledges = new ArrayList<>(featureEdges);
        this.nnodes   = allnodes.size();
        this.tensor   = new ArrayList<>(1 << nnodes);

        // 这里对 hasUNKoperator 没有任何作用，但设为 false 更干净
        this.hasUNKoperator = false;

        // 一共 2^k 个状态，mask 的第 bit 位代表第 bit 个 featureNode 的布尔值
        int totalStates = 1 << nnodes;
        for (int mask = 0; mask < totalStates; mask++) {
            boolean allCorrect = true;
            for (int bit = 0; bit < nnodes; bit++) {
                // 注意：我们沿用现有实现的约定：bit = 1 → 该节点取值 true（“正确”）
                boolean val = ((mask >> bit) & 1) == 1;
                if (!val) {
                    allCorrect = false;
                    break;
                }
            }
            double w = allCorrect ? allCorrectWeight : notAllCorrectWeight;
            this.tensor.add(w);
        }
    }

	public List<Node> getpunodes() {
		ArrayList<Node> ret = new ArrayList<>();
		ret.addAll(preds);
		ret.addAll(uses);
		return ret;
	}

	public Node getstmt() {
		return this.stmt;
	}

	private void gettensor(List<Node> allnodes, int cur) {
		if (cur < 0) {
			tensor.add(getProb());
			return;
		}
		allnodes.get(cur).setTemp(false);
		gettensor(allnodes, cur - 1);
		allnodes.get(cur).setTemp(true);
		gettensor(allnodes, cur - 1);
	}

	// private void ap_gettensor(List<Node> allnodes, int cur) {
	// 	if (cur < 0) {
	// 		ap_tensor.add(ap_getProb());
	// 		return;
	// 	}
	// 	allnodes.get(cur).setTemp(false);
	// 	ap_gettensor(allnodes, cur - 1);
	// 	allnodes.get(cur).setTemp(true);
	// 	ap_gettensor(allnodes, cur - 1);
	// }

	public void send_message() {
		if (!use_ap) {
			// used to save all the messages from the nodes
			List<Double> tmpvlist = new ArrayList<>();
			for (int i = 0; i < nnodes; i++) {
				tmpvlist.add(alledges.get(i).get_ntof());
			}
			// System.out.println("tmplist = "+tmpvlist);
			for (int j = 0; j < nnodes; j++) {
				double v0 = 0;
				double v1 = 0;
				int step = (1 << j);
				int vnum = (1 << nnodes);
				// transform a tensor of nnodes-dimension into a one-dimension vector(two
				// values)
				for (int k = 0; k < vnum; k += 2 * step) {
					for (int o = 0; o < step; o++) {
						int index0 = k + o;
						double tmp0 = tensor.get(index0);

						int index1 = k + o + step;
						double tmp1 = tensor.get(index1);
						// get the bit and times the Corresponding message
						for (int mm = 0; mm < nnodes; mm++) {
							int bit0 = index0 % 2;
							index0 /= 2;
							int bit1 = index1 % 2;
							index1 /= 2;

							if (mm == j)
								continue;

							if (bit0 == 0) {
								// double tmp00 = tmp0* (1 - tmpvlist.get(mm));
								// if(Double.isNaN(tmp00))
								// System.out.println("in 0 , tmp0 = "+tmp0+", val = "+(1 - tmpvlist.get(mm)));
								tmp0 *= (1 - tmpvlist.get(mm));

							} else {
								// double tmp01 = tmp0* tmpvlist.get(mm);
								// if(Double.isNaN(tmp01))
								// System.out.println("in 1 , tmp0 = "+tmp0+", val = "+tmpvlist.get(mm));
								tmp0 *= tmpvlist.get(mm);
							}

							if (bit1 == 0) {
								// double tmp10 = tmp1* (1 - tmpvlist.get(mm));
								// if(Double.isNaN(tmp10))
								// System.out.println("in 0 , tmp1 = "+tmp1+", val = "+(1 - tmpvlist.get(mm)));
								tmp1 *= (1 - tmpvlist.get(mm));

							} else {
								// double tmp11 = tmp1*tmpvlist.get(mm);
								// if(Double.isNaN(tmp11))
								// System.out.println("in 0 , tmp1 = "+tmp1+", val = "+tmpvlist.get(mm));
								tmp1 *= tmpvlist.get(mm);
							}
						}

						v0 += tmp0;
						v1 += tmp1;
					}
				}
				alledges.get(j).set_fton(v1 / (v1 + v0));
			}
		}
		else{

		}
	}

	private void change_parameters(){
		StmtNode tstmt = (StmtNode)stmt;
		int tform = tstmt.form;
		//boolean special = false;
		//for (int i : spopcodes){
		//	if (tform == i)
		//		special = true;
		//}
		int special = 0;
		for (int group=0; group < spopcodes.length; group++){
			for (int i: spopcodes[group]){
				if (tform == i)
					special = group + 1; //0 for determined instruction
			}
			if(special != 0)
				break;
		}
		MEDIUM_LOW = numbersArray.get(special);
		//MEDIUM_LOW = numbersArray2.get(special);
		MEDIUM_HIGH = 1-MEDIUM_LOW;	
	}

	public double getProb() {
		// boolean hasUNKoperator = false;
		// if (ops != null)
		// 	for (String op : ops) {
		// 		for (String unk : unkops) {
		// 			if (op.contentEquals(unk))
		// 				hasUNKoperator = true;
		// 		}
		// 	}
		// if(hasUNKoperator)return MEDIUM;
		// hasUNKoperator = false;
		boolean defv = def.getCurrentValue();
		boolean predv = true;
		boolean usev = true;
		boolean stmtv = stmt.getCurrentValue();

		if (preds != null)
			for (Node p : preds) {
				if (!p.getCurrentValue()) {
					predv = false;
					break;
				}
			}
		if (uses != null)
			for (Node u : uses) {
				if (!u.getCurrentValue()) {
					usev = false;
					break;
				}
			}
		boolean pu = predv && usev;
		if (stmtv) {// if the statement is written correctly.
			if (defv && pu)
				return HIGH;
			if (!defv && !pu) {
				if (hasUNKoperator)
					return MEDIUM_HIGH;
				return HIGH;
			}
			if (!defv && pu)
				return LOW;
			if (defv && !pu) {
				if (hasUNKoperator)
					return MEDIUM_LOW;
				return LOW;
			}
		} else {
			if (defv && pu) {
				if (hasUNKoperator)
					return MEDIUM_LOW;
				return LOW;
			}
			if (!defv && !pu) {
				if (hasUNKoperator)
					return MEDIUM_HIGH;
				return HIGH;
			}
			if (!defv && pu) {
				if (hasUNKoperator)
					return MEDIUM_HIGH;
				return HIGH;
			}
			if (defv && !pu) {
				if (hasUNKoperator)
					return MEDIUM_LOW;
				return LOW;
			}

		}
		return MEDIUM;
	}

	public void print(MyWriter lgr) {

		stmt.print(lgr, "Statement: ");
		if (def != null) {
			lgr.writeln("\tdef:");
			def.print(lgr, "\t\t");
		} else {
			lgr.writeln("\tstmtvalue = " + this.stmtvalue);
		}

		if (uses != null) {
			lgr.writeln("\tuses:");
			for (Node n : uses) {
				n.print(lgr, "\t\t");
			}
		}
		if (preds != null) {
			lgr.writeln("\tpreds:");
			for (Node n : preds) {
				n.print(lgr, "\t\t");
			}
		}
		if (ops != null) {
			lgr.writeln("\tops:");
			for (String eachop : ops) {
				lgr.writeln("\t\t{}", eachop);
			}
		}
	}

	public void print() {
		stmt.print("Statement: ");
		if (def != null) {
			debugLogger.writeln("\tdef:");
			def.print("\t\t");
		} else {
			debugLogger.writeln("\tstmtvalue = " + this.stmtvalue);
		}

		if (uses != null) {
			debugLogger.writeln("\tuses:");
			for (Node n : uses) {
				n.print("\t\t");
			}
		}
		if (preds != null) {
			debugLogger.writeln("\tpreds:");
			for (Node n : preds) {
				n.print("\t\t");
			}
		}
		if (ops != null) {
			debugLogger.writeln("\tops:");
			for (String eachop : ops) {
				debugLogger.writeln("\t\t{}", eachop);
			}
		}
	}
}
