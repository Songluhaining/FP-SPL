package ppfl.instrumentation;

import java.io.IOException;
import java.io.Writer;

import javassist.ClassPool;
import javassist.CtClass;
import javassist.NotFoundException;
import javassist.bytecode.ConstPool;

public class CallBackIndex {

    /**
     * 回调方法名
     */
    private static final String PRINT_CALLBACK_NAME = "printTopStack1";

    // 是否处于 EvoSuite 环境：通过类是否存在来判断
    private static final boolean EVO_ENV;

    static {
        boolean evo = false;
        try {
            Class.forName("org.evosuite.runtime.mock.java.io.NativeMockedIO");
            evo = true;
        } catch (Throwable t) {
            evo = false;
        }
        EVO_ENV = evo;
    }

    // 用于 Elevator / Defects4J 正常场景的文件 writer（TraceTransformer 注入）
    private static volatile Writer writer = null;

    public int logstringindex;
    public int tsindex_int;
    public int tsindex_short;
    public int tsindex_byte;
    public int tsindex_char;
    public int tsindex_boolean;
    public int tsindex_long;
    public int tsindex_float;
    public int tsindex_double;
    public int tsindex_string;
    public int tsindex_object;

    // threshold
    static int loglimit = 1200000;
    static int logcount = 0;

    // ========== 1. 提供给 TraceTransformer 显式注入的 writer（Elevator 等环境用） ==========
    public static void setWriter(Writer w) {
        if (w == null) {
            return;
        }
        writer = w;
    }

    public CallBackIndex(ConstPool constp, Writer w) throws NotFoundException {
        // Elevator / Defects4J 场景下由 TraceTransformer 传进来的 writer
        if (w != null) {
            setWriter(w);
        }
        ClassPool cp = ClassPool.getDefault();
        CtClass thisKlass = cp.get("ppfl.instrumentation.CallBackIndex");
        int classindex = constp.addClassInfo(thisKlass);

        logstringindex = constp.addMethodrefInfo(classindex, "logString", "(Ljava/lang/String;)V");
        tsindex_int = constp.addMethodrefInfo(classindex, PRINT_CALLBACK_NAME, "(I)I");
        tsindex_long = constp.addMethodrefInfo(classindex, PRINT_CALLBACK_NAME, "(J)J");
        tsindex_double = constp.addMethodrefInfo(classindex, PRINT_CALLBACK_NAME, "(D)D");
        tsindex_short = constp.addMethodrefInfo(classindex, PRINT_CALLBACK_NAME, "(S)S");
        tsindex_byte = constp.addMethodrefInfo(classindex, PRINT_CALLBACK_NAME, "(B)B");
        tsindex_char = constp.addMethodrefInfo(classindex, PRINT_CALLBACK_NAME, "(C)C");
        tsindex_boolean = constp.addMethodrefInfo(classindex, PRINT_CALLBACK_NAME, "(Z)Z");
        tsindex_float = constp.addMethodrefInfo(classindex, PRINT_CALLBACK_NAME, "(F)F");
        tsindex_string = constp.addMethodrefInfo(classindex, PRINT_CALLBACK_NAME,
                "(Ljava/lang/String;)Ljava/lang/String;");
        tsindex_object = constp.addMethodrefInfo(classindex, PRINT_CALLBACK_NAME,
                "(Ljava/lang/Object;)Ljava/lang/Object;");
    }

    // ========== 2. LDC 类型选择：保持原逻辑 ==========
    public int getLdcCallBack(Object o) {
        if (o instanceof String)
            return tsindex_string;
        else if (o instanceof Short)
            return tsindex_short;
        else if (o instanceof Long)
            return tsindex_long;
        else if (o instanceof Integer)
            return tsindex_int;
        else if (o instanceof Byte)
            return tsindex_byte;
        else if (o instanceof Character)
            return tsindex_char;
        else if (o instanceof Boolean)
            return tsindex_boolean;
        else if (o instanceof Float)
            return tsindex_float;
        else if (o instanceof Double)
            return tsindex_double;
        else
            return tsindex_object;
    }

    // ========== 3. 底层输出工具：根据是否 EvoSuite 选择输出位置 ==========

    private static void writeStack(String prefix, String value) {
        if (EVO_ENV) {
            // EvoSuite 环境下：绕开 FileWriter，直接打到标准输出
            System.out.print(prefix);
            if (value != null) {
                System.out.print(value);
            }
            System.out.flush();
        } else {
            // 普通环境：用 TraceTransformer 注入的 writer 写文件
            Writer w = writer;
            if (w == null) {
                return;
            }
            try {
                w.write(prefix);
                if (value != null) {
                    w.write(value);
                }
                w.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static void writeLogString(String s) {
        if (EVO_ENV) {
            System.out.print(s);
            System.out.flush();
        } else {
            Writer w = writer;
            if (w == null) {
                return;
            }
            try {
                w.write(s);
                w.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    // ========== 4. 各种 printTopStack1：保持语义 + 调用统一输出 ==========
    public static int printTopStack1(int i) {
        writeStack(",stack=I:", String.valueOf(i));
        return i;
    }

    public static double printTopStack1(double i) {
        writeStack(",stack=D:", String.valueOf(i));
        return i;
    }

    public static short printTopStack1(short i) {
        writeStack(",stack=S:", String.valueOf(i));
        return i;
    }

    public static char printTopStack1(char i) {
        // FIXME: 原实现也没有真正写 char 值，这里保持一致行为
        writeStack(",stack=C:", null);
        return i;
    }

    public static byte printTopStack1(byte i) {
        writeStack(",stack=B:", String.valueOf(i));
        return i;
    }

    public static boolean printTopStack1(boolean i) {
        writeStack(",stack=Z:", String.valueOf(i));
        return i;
    }

    public static float printTopStack1(float i) {
        writeStack(",stack=F:", String.valueOf(i));
        return i;
    }

    public static long printTopStack1(long i) {
        writeStack(",stack=J:", String.valueOf(i));
        return i;
    }

    public static String printTopStack1(String i) {
        // FIXME: 原实现也没有真正写字符串内容，这里保持一致
        writeStack(",stack=Str:", null);
        return i;
    }

    public static Object printTopStack1(Object i) {
        String val = (i == null) ? "0" : String.valueOf(java.lang.System.identityHashCode(i));
        writeStack(",stack=Obj:", val);
        return i;
    }

    // ========== 5. logString / flush：同样区分 EvoSuite / 非 EvoSuite ==========
    public static void logString(String s) {
        logcount++;
        if (logcount > loglimit) {
            System.exit(0);
        }
        writeLogString(s);
    }

    public static void flush() {
        if (EVO_ENV) {
            System.out.flush();
        } else {
            Writer w = writer;
            if (w == null) {
                return;
            }
            try {
                w.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

}
