package ppfl.instrumentation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.IllegalClassFormatException;
import java.nio.file.Paths;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

// import org.slf4j.Logger;
// import org.slf4j.LoggerFactory;
// import org.slf4j.MDC;

import javassist.ClassPool;
import javassist.CtBehavior;
import javassist.CtClass;
import javassist.NotFoundException;
import javassist.bytecode.BadBytecode;
import javassist.bytecode.CodeAttribute;
import javassist.bytecode.CodeIterator;
import javassist.bytecode.ConstPool;
import javassist.bytecode.MethodInfo;
import javassist.bytecode.Mnemonic;
import ppfl.MyWriter;
import ppfl.ProfileUtils;
import ppfl.WriterUtils;
import ppfl.instrumentation.opcode.InvokeInst;
import ppfl.instrumentation.opcode.OpcodeInst;
import javassist.bytecode.LineNumberAttribute;

public class TraceTransformer implements ClassFileTransformer {

	private boolean useCachedClass = true;

	private static MyWriter debugLogger = WriterUtils.getWriter("Debugger_trace");
	// LoggerFactory.getLogger(TraceTransformer.class);

	// The logger name
	public static final String TRACELOGGERNAME = "PPFL_LOGGER";
	public static final String SOURCELOGGERNAME = "PPFL_LOGGER_SOURCE";
	// public static final Logger traceLogger =
	// LoggerFactory.getLogger(TRACELOGGERNAME);
	// public static final Logger sourceLogger =
	// LoggerFactory.getLogger(SOURCELOGGERNAME);
	private static final int BUFFERSIZE = 1 << 20;
	private static BufferedWriter sourceWriter = null;
	private static BufferedWriter staticInitWriter = null;
	private static Writer traceWriter = null;
	private static BufferedWriter whatIsTracedWriter = null;
	/** The internal form class name of the class to transform */
	private String targetClassName;
	/** The class loader of the class we want to transform */
	private ClassLoader targetClassLoader;

	private boolean useD4jTest = false;
	private Set<String> d4jMethodNames = new HashSet<>();
	private boolean logSourceToScreen = false;
	private boolean simpleLog = false;

	private Set<String> transformedMethods = new HashSet<>();

	/** filename for logging */
	public TraceTransformer(String targetClassName, ClassLoader targetClassLoader, String logfilename) {
		this.targetClassName = targetClassName;
		this.targetClassLoader = targetClassLoader;

		File logdir = new File("trace/logs/run/");
		logdir.mkdirs();
		Interpreter.init();
		setWhatIsTracedWriterFile();
		FileWriter file = null;
		try {
			file = new FileWriter("trace/logs/run/" + logfilename, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		traceWriter = file;
		CallBackIndex.setWriter(traceWriter);
		// traceWriter = new BufferedWriter(file, BUFFERSIZE);
		Runtime.getRuntime().addShutdownHook(new Thread() {
			@Override
			public void run() {
				try {
					TraceTransformer.traceWriter.flush();
					// closing the stream may trigger double-close bug.
					// TraceTransformer.traceWriter.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		});
	}

	private static void setStaticInitFile(String clazzname) {
		setStaticInitWriterFile(String.format("trace/logs/mytrace/%s.init.log", clazzname));
	}

	private static void setStaticInitWriterFile(String filename) {
		FileWriter file = null;
		try {
			file = new FileWriter(filename);
		} catch (IOException e) {
			e.printStackTrace();
		}
		staticInitWriter = new BufferedWriter(file, BUFFERSIZE);
	}

	private static void setSourceFile(String clazzname) {
		setWriterFile(String.format("trace/logs/mytrace/%s.source.log", clazzname));
	}

	private static void setWriterFile(String filename) {
		FileWriter file = null;
		try {
			file = new FileWriter(filename);
		} catch (IOException e) {
			e.printStackTrace();
		}
		sourceWriter = new BufferedWriter(file, BUFFERSIZE);
	}

	private static void setWhatIsTracedWriterFile() {
		FileWriter file = null;
		String filename = "trace/logs/mytrace/traced.source.log";
		try {
			File logdir = new File("trace/logs/mytrace/");
			logdir.mkdirs();
			file = new FileWriter(filename, true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		whatIsTracedWriter = new BufferedWriter(file, BUFFERSIZE);
	}

	public void setLogSourceToScreen(boolean b) {
		this.logSourceToScreen = b;
	}

	private static void setSimpleLogFile() {
		FileWriter file = null;
		try {
			File logdir = new File("trace/logs/mytrace/");
			logdir.mkdirs();
			file = new FileWriter("trace/logs/mytrace/profile.log", true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		traceWriter = file;
		// traceWriter = new BufferedWriter(file, BUFFERSIZE);
	}

	public void setSimpleLog(boolean b) {
		this.simpleLog = b;
		setSimpleLogFile();
	}

	public void setLogFile(String s) {
		// String logFile = null;
		// logFile = s.replace('\\', '.').replace('/', '.');
		// MDC.put("logfile", logFile);
	}

	public void setD4jDataFile(String filepath) {
		// System.out.println(filepath);
		useD4jTest = true;
		String methodstring = "methods.test.all=";
		try (BufferedReader reader = new BufferedReader(new FileReader(filepath))) {
			String s = null;
			while ((s = reader.readLine()) != null) {
				if (!s.startsWith(methodstring))
					continue;
				String[] classandmethods = s.substring(methodstring.length()).split(";");
				// Collections.addAll(d4jMethodNames, methodnames);
				for (String tmp : classandmethods) {
					if (!tmp.isEmpty()) {
						String[] splt = tmp.split("::");
						if (splt.length < 2)
							continue;
						String[] methodsname = splt[1].split(",");
						for (String methodname : methodsname) {
							d4jMethodNames.add(splt[0] + "::" + methodname);
						}
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		// System.out.println(d4jMethodNames.size());
		ProfileUtils.setD4jMethods(d4jMethodNames);
	}

	private boolean isD4jTestMethod(CtClass cc, CtBehavior m) {
		if (!useD4jTest) {
			return false;
		}
		String longname = cc.getName() + "::" + m.getName();
		return d4jMethodNames.contains(longname);
	}

	private void writeWhatIsTraced(String str) {
		if (this.simpleLog)
			return;
		try {
			whatIsTracedWriter.write(str);
			whatIsTracedWriter.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	protected byte[] transformBody(String classname) {
    byte[] byteCode = null;
    // 统一用点号形式
    classname = classname.replace("/", ".");
    // 日志里也用真正的类名，避免 targetClassName 偶尔不一致
    debugLogger.write(String.format("[Agent] Transforming class %s", classname));

    // ===================== 1. 读取缓存（若开启） =====================
    if (useCachedClass) {
        String classcachefolder = "trace/classcache/";
        File folder = new File(classcachefolder);
        if (!folder.exists()) {
            folder.mkdirs();
        }
        File classcache = new File(folder, classname + ".log");
        if (!this.simpleLog && classcache.exists()) {
            try {
                byte[] cached = java.nio.file.Files.readAllBytes(classcache.toPath());
                // 防御：缓存文件存在但内容为空时，不要继续用它
                if (cached != null && cached.length > 0) {
                    return cached;
                }
            } catch (IOException e) {
                e.printStackTrace();
                // 读缓存失败就当没缓存，用正常插桩流程
            }
        }
    }

    // ===================== 2. 非 simpleLog 下初始化 logger 等 =====================
    if (!this.simpleLog) {
        this.setLogger(classname);
        setSourceFile(classname);
        setStaticInitFile(classname);
    }

    // ===================== 3. 真正的插桩逻辑 =====================
    try {
        ClassPool cp = ClassPool.getDefault();
        CtClass cc = cp.get(classname);

        if (!this.simpleLog) {
            MethodInfo staticInit = cc.getClassFile().getStaticInitializer();
            if (staticInit != null) {
                getStaticInitializerInfo(staticInit, cc);
            }
        }

        if (!cc.getClassFile().getMethods().isEmpty() || cc.getClassFile().getSuperclass() != null) {
            writeWhatIsTraced("\n" + classname + "::");
        }

        boolean instrumentJunit = true; // evaluation switch
        for (MethodInfo m : cc.getClassFile().getMethods()) {
            // 跳过 junit 自身大部分方法
            if (instrumentJunit && cc.getName().startsWith("junit") && !m.getName().startsWith("assert")) {
                continue;
            }
            if (!m.isStaticInitializer()) {
                writeWhatIsTraced(m.getName() + "#" + m.getDescriptor() + ",");
                transformBehavior(m, cc);
            }
        }

        // dump class inheritance
        String superClassName = cc.getClassFile().getSuperclass();
        if (superClassName != null) {
            writeWhatIsTraced(superClassName + "#" + "SuperClass");
        }

        // 生成插桩后的字节码
        byteCode = cc.toBytecode();
        cc.detach();

    } catch (Exception e) {
        // 这里非常关键：插桩失败时不要继续往下写缓存，更不要让 JVM 崩掉
        System.err.println("[Agent] bytecode transform failed for " + classname);
        e.printStackTrace();
        // 告诉 Instrumentation：“我放弃修改这个类，让你用原始字节码吧”
        return null;
    }

    // ===================== 4. 写缓存（若开启） =====================
    if (!this.simpleLog && useCachedClass) {
        if (byteCode != null) {
            try {
                String classcachefolder = "trace/classcache/";
                java.nio.file.Files.write(Paths.get(classcachefolder, classname + ".log"), byteCode);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            // 按理说走到这里 byteCode 不应该是 null，防御性信息打印
            System.err.println("[Agent] byteCode is null after transform for " + classname + ", skip caching.");
        }
    }

    return byteCode;
}

	private void getStaticInitializerInfo(MethodInfo m, CtClass cc) throws BadBytecode {
		MethodInfo mi = m;
		CodeAttribute ca = mi.getCodeAttribute();

		ConstPool constp = mi.getConstPool();
		CodeIterator tempci = ca.iterator();
		StringBuilder sb = new StringBuilder();
		for (int i = 0; tempci.hasNext(); i++) {
			int index = tempci.lookAhead();
			int ln = mi.getLineNumber(index);
			String getinst = getInstMap(tempci, index, constp);
			String sig = mi.getDescriptor();
			// ExceptionTable eTable =
			// m.getMethodInfo().getCodeAttribute().getExceptionTable();
			String linenumberinfo = ",lineinfo=" + cc.getName() + "#" + mi.getName() + "#" + sig + "#" + ln + "#" + index
					+ ",nextinst=";

			tempci.next();
			if (!tempci.hasNext()) {
				linenumberinfo = linenumberinfo + "-1";
			} else {
				linenumberinfo = linenumberinfo + String.valueOf(tempci.lookAhead());
			}
			String instinfo = getinst + linenumberinfo;
			sb.append(instinfo);
		}
		try {
			staticInitWriter.write(sb.toString());
			staticInitWriter.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

    private void transformBehavior(MethodInfo m, CtClass cc) throws NotFoundException, BadBytecode {
        // 这里的 m 就是当前方法的 MethodInfo
        MethodInfo mi = m;
        CodeAttribute ca = mi.getCodeAttribute();
        // 对于 abstract / native 方法，可能没有 CodeAttribute，直接跳过
        if (ca == null) {
            return;
        }

        // 常量池 & 回调索引
        ConstPool constp = mi.getConstPool();
        CallBackIndex cbi = new CallBackIndex(constp, traceWriter);

        // ===========================
        // 1. 字节码级插桩（详细 trace 模式）
        // ===========================
        if (!this.simpleLog) {
            // 对方法体逐条指令插桩（内部会往 traceWriter / sourceWriter 写日志）
            instrumentByteCode(cc, mi, ca, constp, cbi);

            // 在方法入口处插入一条日志，打印 "###Class::method"
            CodeIterator ci = ca.iterator();
            String longname = String.format("%n###%s::%s", cc.getName(), mi.getName());
            int instpos   = ci.insertGap(6);
            int instindex = constp.addStringInfo(longname);

            // ldc_w <longname>
            ci.writeByte(19, instpos);          // 19 == Opcode.LDC_W
            ci.write16bit(instindex, instpos + 1);
            // invokestatic CallBack.logString(...)
            ci.writeByte(184, instpos + 3);     // 184 == Opcode.INVOKESTATIC
            ci.write16bit(cbi.logstringindex, instpos + 4);
        }
        // ===========================
        // 2. simpleLog 模式：旧的 ProfileUtils 逻辑（SMARTFL 原本就有）
        // ===========================
        else {
            ProfileUtils.init(constp, traceWriter);
            CodeIterator ci = ca.iterator();
            String longname = String.format("%s::%s", cc.getName(), mi.getName());
            ProfileUtils.logMethodName(ci, longname, constp);
        }

        // 重新计算 max stack，保证插桩后的字节码栈深度正确
        ca.computeMaxStack();

        // 刷新 sourceWriter（只刷新，不再写入任何自定义 [LINE_TABLE] 内容）
        if (!this.simpleLog && sourceWriter != null) {
            try {
                sourceWriter.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

	private void instrumentByteCode(CtClass cc, MethodInfo mi, CodeAttribute ca, ConstPool constp, CallBackIndex cbi)
			throws BadBytecode {
		// record line info and instructions, since instrumentation will change
		// branchbyte and byte index.
		CodeIterator tempci = ca.iterator();
		Map<Integer, String> instmap = new HashMap<>();
		for (int i = 0; tempci.hasNext(); i++) {
			int index = tempci.lookAhead();
			int ln = mi.getLineNumber(index);
			String getinst = getInstMap(tempci, index, constp);
			String sig = mi.getDescriptor();
			// ExceptionTable eTable =
			// m.getMethodInfo().getCodeAttribute().getExceptionTable();
			String linenumberinfo = ",lineinfo=" + cc.getName() + "#" + mi.getName() + "#" + sig + "#" + ln + "#" + index
					+ ",nextinst=";

			tempci.next();
			if (!tempci.hasNext()) {
				linenumberinfo = linenumberinfo + "-1";
			} else {
				linenumberinfo = linenumberinfo + String.valueOf(tempci.lookAhead());
			}
			instmap.put(i, getinst + linenumberinfo);
		}
		// iterate every instruction
		CodeIterator ci = ca.iterator();
		for (int i = 0; ci.hasNext(); i++) {
			// lookahead the next instruction.
			int index = ci.lookAhead();
			int op = ci.byteAt(index);
			OpcodeInst oi = Interpreter.map[op];
			// linenumber information.
			String instinfo = instmap.get(i);

			// insert bytecode right before this inst.
			// print basic information of this instruction
			// if (logSourceToScreen)
			// sourceLogger.info(instinfo);

			try {
				sourceWriter.write(instinfo);
			} catch (IOException e) {
				e.printStackTrace();
			}

			if (oi != null) {
				if (!mi.isStaticInitializer())
					oi.insertByteCodeBefore(ci, index, constp, instinfo, cbi);
			}
			int previndex = index;
			// move to the next inst. everything below this will be inserted after the inst.
			index = ci.next();
			// print advanced information(e.g. value pushed)
			if (oi != null) {
				// if (oi.form > 42)
				// getstatic should be treated like invocation,
				// in the case that static-initializer may be called.
				if (oi instanceof InvokeInst || oi.form == 178 || oi.form == 187) {// getstatic and new
					if (!mi.isStaticInitializer())
						oi.insertReturnSite(ci, index, constp, instinfo, cbi);
				} else {
					if (!mi.isStaticInitializer())
						oi.insertByteCodeAfter(ci, index, constp, cbi);
				}
			}
		}
	}

	@Override
    public byte[] transform(ClassLoader loader, String className, Class<?> classBeingRedefined,
                            ProtectionDomain protectionDomain, byte[] classfileBuffer) throws IllegalClassFormatException {
        try {
            byte[] byteCode = classfileBuffer;

            boolean hit = false;
            if (className != null) {
                String[] targets = this.targetClassName.split(":");
                for (String t : targets) {
                    t = t.trim();
                    if (t.isEmpty()) continue;
                    String finalTargetClassName = t.replace(".", "/");
                    if (className.equals(finalTargetClassName)) {
                        hit = true;
                        break;
                    }
                }
            }

            // 只按类名过滤；不再限制 loader 必须等于某一个 targetClassLoader
            if (!hit || loader == null) {
                return byteCode;
            }

//            System.out.println("[DEBUG-transform] className = " + className
//                    + ", loader=" + loader.getClass().getName());

            // 可选：更新 targetClassLoader，方便以后调试/使用
            this.targetClassLoader = loader;

            return transformBody(className);

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

	private void setLogger(String clazzname) {
		// MDC.put("sourcefile", clazzname);
	}

	private String getInstMap(CodeIterator ci, int index, ConstPool constp) {
		int op = ci.byteAt(index);
		String inst = null;
		String opc = Mnemonic.OPCODE[op];
		OpcodeInst oi = Interpreter.map[op];
		if (oi == null) {
			debugLogger.write("unsupported opcode: ");
			debugLogger.write(opc + "\n");
			return "";
		}
		inst = oi.getinst(ci, index, constp);
		return inst;
	}
}