package activation;

public class ActivationSigmoid implements ActivationFunction{
    @Override
    public double activation(double d) {
        return 1.0 / (1 + Math.exp(-1.0 * d));
    }

    @Override
    public double derivative(double d) {
        return d * (1 - d);
    }
}
