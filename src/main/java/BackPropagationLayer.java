import math.Matrix;

public class BackPropagationLayer {
    private double[] error;

    private double[] errorDelta;

    private Matrix accumulateMatrixDelta;

    // Hold the previous matrix deltas so that "momentum" can be implemented.
    private Matrix matrixDelta;

    // The index of bias location
    private int biasRow;

    // Parent
    private final BackPropagation backPropagation;

    private final FeedforwardLayer feedforwardLayer;

    public BackPropagationLayer(BackPropagation backPropagation, FeedforwardLayer feedforwardLayer) {
        this.backPropagation = backPropagation;
        this.feedforwardLayer = feedforwardLayer;

        int neuronCount = feedforwardLayer.getNeuronCount();

        this.error = new double[neuronCount];
        this.errorDelta = new double[neuronCount];

        // If it's not an output layer
        if (!feedforwardLayer.isOutput()) {
            this.accumulateMatrixDelta = new Matrix(neuronCount + 1, feedforwardLayer
                    .getNext().getNeuronCount());
            this.matrixDelta = new Matrix(neuronCount + 1, feedforwardLayer
                    .getNext().getNeuronCount());
            this.biasRow = neuronCount;
        }
    }

    /**
     * Calculate the error for other layer
     */
    public void calcError() {
        BackPropagationLayer next = backPropagation.getBackPropagationLayer(this.feedforwardLayer.getNext());

        int neuronCount = feedforwardLayer.getNext().getNeuronCount();
        for (int i = 0; i < neuronCount; i++) {
            for (int j = 0; j < feedforwardLayer.getNeuronCount(); j++) {
                accumulateMatrixDelta(j, i, next.getErrorDelta(i)
                        * feedforwardLayer.getValue(j));
                error[j] = getError(j) + feedforwardLayer.getMatrix().get(j, i)
                        * next.getErrorDelta(i);
            }
            accumulateThresholdDelta(i, next.getErrorDelta(i));
        }

        if (feedforwardLayer.isHidden()) {
            for (int i = 0; i < feedforwardLayer.getNeuronCount(); i++) {
                errorDelta[i] = calculateDelta(error[i], feedforwardLayer.getValue(i));
            }
        }
    }

    /**
     * Calculate the error for the output layer
     *（expected - output) * derivative(output)
     *
     * @param expected the expected result
     */
    public void calcError(double[] expected) {
        int neuronCount = feedforwardLayer.getNeuronCount();
        ;
        for (int i = 0; i < neuronCount; i++) {
            error[i] = expected[i] - feedforwardLayer.getValue(i);
            errorDelta[i] = calculateDelta(error[i], feedforwardLayer.getValue(i));
        }
    }

    private void accumulateThresholdDelta(int i, double errorDelta) {
        accumulateMatrixDelta.add(biasRow, i, errorDelta);
    }

    private double getError(int j) {
        return error[j];
    }

    private void accumulateMatrixDelta(int i1, int i2, double v) {
        accumulateMatrixDelta.add(i1, i2, v);
    }

    private double calculateDelta(double error, double output) {
        return error * feedforwardLayer.getActivationFunction().derivative(output);
    }

    public double getErrorDelta(int i) {
        return errorDelta[i];
    }

    public void clearError() {
        for (int i = 0; i < feedforwardLayer.getNeuronCount(); i++) {
            error[i] = 0;
        }
    }

    public void learn(double learningRate, double momentum) {
        // process the matrix
        if (feedforwardLayer.hasMatrix()) {
            final Matrix m1 = accumulateMatrixDelta.multiply(learningRate);
            final Matrix m2 = matrixDelta.multiply(momentum);
            matrixDelta = m1.add(m2);
            feedforwardLayer.setMatrix(feedforwardLayer.getMatrix().add(this.matrixDelta));
            accumulateMatrixDelta.clear();
        }
    }
}
