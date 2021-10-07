package activation;

public interface ActivationFunction {
    /**
     * A activation function for a neural network.
     * @param d The input to the function.
     * @return The output from the function.
     */
    public double activation(double d);

    /**
     * Performs the derivative of the activation function function on the input.
     *
     * @param d The input.
     * @return The output.
     */
    public double derivative(double d);
}
