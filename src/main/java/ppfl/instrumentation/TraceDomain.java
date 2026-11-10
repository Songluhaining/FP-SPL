package ppfl.instrumentation;

/**
 * 关键优化点：
 * - toString() 不再每次拼接新字符串，而是一次构造、永久复用（cachedToString）。
 * - hashCode 也提前算好（cachedHash），避免反复计算。
 *
 * 语义保持完全一致：equals/hashCode 逻辑不变，compareCallClass 开关也不变。
 */
public class TraceDomain {
    public final String traceclass;
    public final String tracemethod;
    public final String signature;

    static boolean compareCallClass = true; // evaluation switch

    // ===== 缓存：避免高频 toString()/hashCode() 产生大量临时对象 =====
    private final String cachedToString;
    private final int cachedHash;

    public TraceDomain(String traceclass, String tracemethod, String signature) {
        // trim 仍然保留，保证与原行为一致
        this.traceclass = (traceclass == null) ? "" : traceclass.trim();
        this.tracemethod = (tracemethod == null) ? "" : tracemethod.trim();
        this.signature = (signature == null) ? "" : signature.trim();

        // 一次性拼接：之后 toString() 直接返回同一个 String 对象
        this.cachedToString = this.traceclass + ":" + this.tracemethod + ":" + this.signature;

        // 一次性计算 hash：与原 hashCode() 公式完全一致
        int result = this.tracemethod.hashCode();
        if (compareCallClass) {
            result = result * 31 + this.traceclass.hashCode();
        }
        result = result * 31 + this.signature.hashCode();
        this.cachedHash = result;
    }

    @Override
    public String toString() {
        return this.cachedToString;
    }

    @Override
    public boolean equals(Object oth) {
        if (oth == null || !(oth instanceof TraceDomain)) return false;
        if (this == oth) return true;
        TraceDomain instance = (TraceDomain) oth;

        if (compareCallClass) {
            return traceclass.equals(instance.traceclass)
                    && tracemethod.equals(instance.tracemethod)
                    && signature.equals(instance.signature);
        }
        return tracemethod.equals(instance.tracemethod)
                && signature.equals(instance.signature);
    }

    @Override
    public int hashCode() {
        return this.cachedHash;
    }
}
