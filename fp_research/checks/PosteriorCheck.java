package ppfl;

/** A unary factor with prior 0.8 has exact posterior correctness 0.8. */
public class PosteriorCheck {
    public static void main(String[] args) {
        Node node = new Node("example");
        Edge edge = new Edge();
        edge.setnode(node);
        FactorNode prior = new FactorNode(node, edge, 0.8);
        edge.setfactor(prior);
        node.add_edge(edge);
        prior.send_message();
        node.send_message();
        double suspect = 1.0 - node.bp_getprob();
        if (Math.abs(suspect - 0.2) > 1e-9) {
            throw new AssertionError(suspect);
        }
        System.out.println(suspect);
    }
}
