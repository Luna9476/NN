import java.util.HashMap;
import java.util.Map;

public class BackPropagation {
    private final double learningRate;

    private final double momentum;

    private final FeedforwardNetwork network;

    private Map<FeedforwardLayer, BackPropagationLayer> map = new HashMap<>();

    private double[][] input;

    private double[][] expected;


    public BackPropagation(double learningRate, double momentum, double[][] input, double[][] expected, FeedforwardNetwork feedforwardNetwork) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.input = input;
        this.expected = expected;
        this.network = feedforwardNetwork;

        for (FeedforwardLayer layer : feedforwardNetwork.getLayers()) {
            BackPropagationLayer bpl = new BackPropagationLayer(this, layer);
            map.put(layer, bpl);
        }
    }

    public double train() {
        double error = 0;
        for (int i = 0; i < input.length; i ++) {
            // forward
            double[] outputs = network.computeOutputs(input[i]);
            // compute the total error
            for (int j = 0; j < outputs.length; j++) {
                error += Math.pow(outputs[j] - expected[i][j], 2);
            }
            // calculate error
            calcError(expected[i]);
            learn();

        }
        return error;
    }

    /**
     * Calculate error for every layer
     * @param expected the expected output vector
     */
    private void calcError(double[] expected) {
        for (FeedforwardLayer layer : network.getLayers()) {
            getBackPropagationLayer(layer).clearError();
        }

        // backForward from the output layer to calculate the error delta for every layer
        for (int i = network.getLayers().size() - 1; i >= 0; i--) {
            final FeedforwardLayer layer = network.getLayers().get(i);
            if (layer.isOutput()) {
                // output layer
                getBackPropagationLayer(layer).calcError(expected);
            } else {
                // hidden layer
                getBackPropagationLayer(layer).calcError();
            }
        }
    }

    /**
     * Modify the weight matrix and thresholds based on the last call to
     * calcError.
     */
    public void learn() {
        for (final FeedforwardLayer layer : this.network.getLayers()) {
            getBackPropagationLayer(layer).learn(this.learningRate, this.momentum);
        }

    }

    public BackPropagationLayer getBackPropagationLayer(FeedforwardLayer layer) {
        return map.get(layer);
    }
}
