public interface NeuralNetInterface extends CommonInterface{
    final double bias = 1.0;


    /**
     * Return bipolar sigmoid of x
     * @param x the input
     * @return f(x) = 1 /(1+e(-x)) - 1
     */
    public double sigmoid(double x);

    /**
     * A general sigmoid with asymtotes bounded by (a, b)
     * @param x the input
     * @return f(x) = b_minus_a / (1 + e(-x)) - minus_a
     */
    public double customSigmoid(double x, double low, double up);


    /**
     * Initialize the weights to random values
     */
    public void initializeWeights();

    /**
     * Initialize the weights to 0.
     */
    public void zeroWeights();
}
