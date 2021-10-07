import activation.ActivationFunction;
import activation.ActivationSigmoid;
import math.Matrix;

/**
 * FeedforwardLayer: This class represents one layer in a
 * feed forward neural network.
 * This layer could be input, output, or hidden, depending on its placement inside of
 * the FeedforwardNetwork class.
 *
 * An activation function can also be specified.
 * By default this class uses the sigmoid activation function.
 */
public class FeedforwardLayer {
    /**
     * Results from the last time that the outputs were calculated for this layer.
     */
    private double[] values;

    /**
     * The weight and bias matrix.
     */
    private Matrix matrix;

    /**
     * The next layer in the neural network.
     */
    private FeedforwardLayer next;

    /**
     * The previous layer in the neural network.
     */
    private FeedforwardLayer previous;

    /**
     * Which activation function to use for this layer.
     */
    private final ActivationFunction activationFunction;

    /**
     * Construct this layer with a non-default bias function.
     *
     * @param biasFunction The bias function to use.
     * @param neuronCount       How many neurons in this layer.
     */
    public FeedforwardLayer(final ActivationFunction biasFunction,
                            final int neuronCount) {
        this.values = new double[neuronCount];
        this.activationFunction = biasFunction;
    }

    /**
     * Construct this layer with a sigmoid bias function.
     *
     * @param neuronCount How many neurons in this layer.
     */
    public FeedforwardLayer(final int neuronCount) {
        this(new ActivationSigmoid(), neuronCount);
    }

    /**
     * Clone the structure of this layer, but do not copy any matrix data.
     *
     * @return The cloned layer.
     */
    public FeedforwardLayer cloneStructure() {
        return new FeedforwardLayer(this.activationFunction, this.getNeuronCount());
    }

    /**
     * Compute the outputs for this layer given the input pattern.
     * The output is also stored in the fire instance variable.
     *
     * @param pattern The input pattern.
     * @return The output from this layer.
     */
    public void computeOutputs(final double[] pattern) {
        int i;
        // If it's input layer, set the value
        if (pattern != null) {
            for (i = 0; i < getNeuronCount(); i++) {
                setValue(i, pattern[i]);
            }
        }

        final Matrix inputMatrix = createInputMatrix(values);

        for (i = 0; i < this.next.getNeuronCount(); i++) {
            final Matrix col = this.matrix.getCol(i);
            final double sum = col.dotProduct(inputMatrix);
            // compute value for the next layer
            this.next.setValue(i, this.activationFunction.activation(sum));
        }
    }

    /**
     * Take a simple double array and turn it into a matrix that can be used to
     * calculate the results of the input array. Also takes into account the
     * bias.
     *
     * @param pattern
     * @return A matrix that represents the input pattern.
     */
    private Matrix createInputMatrix(final double[] pattern) {
        final Matrix result = new Matrix(1, pattern.length + 1);
        for (int i = 0; i < pattern.length; i++) {
            result.set(0, i, pattern[i]);
        }

        // add a "fake" first column to the input so that the bias is
        // always multiplied by one, resulting in it just being added.
        result.set(0, pattern.length, 1);

        return result;
    }

    /**
     * Get the output array from the last time that the output of this layer was
     * calculated.
     *
     * @return The output array.
     */
    public double[] getValues() {
        return this.values;
    }

    /**
     * Get the output from an individual neuron.
     *
     * @param index The neuron specified.
     * @return The output from the specified neuron.
     */
    public double getValue(final int index) {
        return this.values[index];
    }

    /**
     * Get the weight and bias matrix.
     *
     * @return The weight and bias matrix.
     */
    public Matrix getMatrix() {
        return this.matrix;
    }

    /**
     * Get the size of the matrix, or zero if one is not defined.
     *
     * @return The size of the matrix.
     */
    public int getMatrixSize() {
        if (this.matrix == null) {
            return 0;
        } else {
            return this.matrix.size();
        }
    }

    /**
     * Get the neuron count for this layer.
     *
     * @return the neuronCount
     */
    public int getNeuronCount() {
        return values.length;
    }

    /**
     * @return the next layer.
     */
    public FeedforwardLayer getNext() {
        return this.next;
    }

    /**
     * @return the previous layer.
     */
    public FeedforwardLayer getPrevious() {
        return this.previous;
    }

    /**
     * Determine if this layer has a matrix.
     *
     * @return True if this layer has a matrix.
     */
    public boolean hasMatrix() {
        return this.matrix != null;
    }

    /**
     * Determine if this is a hidden layer.
     *
     * @return True if this is a hidden layer.
     */
    public boolean isHidden() {
        return ((this.next != null) && (this.previous != null));
    }

    /**
     * Determine if this is an input layer.
     *
     * @return True if this is an input layer.
     */
    public boolean isInput() {
        return (this.previous == null);
    }

    /**
     * Determine if this is an output layer.
     *
     * @return True if this is an output layer.
     */
    public boolean isOutput() {
        return (this.next == null);
    }

    /**
     * Prune one of the neurons from this layer. Remove all entries in this
     * weight matrix and other layers.
     *
     * @param neuron The neuron to prune. Zero specifies the first neuron.
     */
//    public void prune(final int neuron) {
//        // delete a row on this matrix
//        if (this.matrix != null) {
//            setMatrix(MatrixMath.deleteRow(this.matrix, neuron));
//        }
//
//        // delete a column on the previous
//        final FeedforwardLayer previous = this.getPrevious();
//        if (previous != null) {
//            if (previous.getMatrix() != null) {
//                previous.setMatrix(MatrixMath.deleteCol(previous.getMatrix(),
//                        neuron));
//            }
//        }
//
//    }

    /**
     * Reset the weight matrix and bias values to random numbers between -1
     * and 1.
     */
    public void reset(double lower, double upper) {

        if (this.matrix != null) {
            this.matrix.randomize(lower, upper);

        }

    }

    /**
     * Set the last output value for the specified neuron.
     *
     * @param index The specified neuron.
     * @param f     The fire value for the specified neuron.
     */
    public void setValue(final int index, final double f) {
        this.values[index] = f;
    }

    /**
     * Assign a new weight and bias matrix to this layer.
     *
     * @param matrix The new matrix.
     */
    public void setMatrix(final Matrix matrix) {
        this.values = new double[matrix.getRows() - 1];
        this.matrix = matrix;

    }

    /**
     * Set the next layer.
     *
     * @param next the next layer.
     */
    public void setNext(final FeedforwardLayer next) {
        this.next = next;
        // add one to the neuron count to provide a bias value in row 0
        this.matrix = new Matrix(this.getNeuronCount() + 1, next
                .getNeuronCount());
    }

    /**
     * Set the previous layer.
     *
     * @param previous the previous layer.
     */
    public void setPrevious(final FeedforwardLayer previous) {
        this.previous = previous;
    }

    /**
     * Produce a string form of the layer.
     */
    @Override
    public String toString() {
        return "[FeedforwardLayer: Neuron Count=" +
                getNeuronCount() +
                "]";
    }

    public ActivationFunction getActivationFunction() {
        return this.activationFunction;
    }

}