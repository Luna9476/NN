import java.util.ArrayList;
import java.util.List;

public class FeedforwardNetwork {
    private List<FeedforwardLayer> layers = new ArrayList<>();

    private FeedforwardLayer inputLayer;
    private FeedforwardLayer outputLayer;


    public FeedforwardNetwork() {

    }

    public void addLayer(FeedforwardLayer layer) {
        // setup the forward and back pointer
        if (this.outputLayer != null) {
            layer.setPrevious(this.outputLayer);
            this.outputLayer.setNext(layer);
        }

        // update the inputLayer and outputLayer variables
        if (this.layers.size() == 0) {
            this.inputLayer = this.outputLayer = layer;
        } else {
            this.outputLayer = layer;
        }

        // add the new layer to the list
        this.layers.add(layer);
    }

    /**
     * Compute the output based on the input value by using neuron layers
     * @param input input value
     * @return output vector
     */
    public double[] computeOutputs(double[] input) {
        for (FeedforwardLayer layer : layers) {
            if (layer.isInput()) {
                layer.computeOutputs(input);
            } else if (layer.isHidden()){
                layer.computeOutputs(null);
            }
        }
        return outputLayer.getValues();
    }

    public List<FeedforwardLayer> getLayers() {
        return layers;
    }

    /**
     * Reset the weight matrix and the bias.
     */
    public void reset(double lower, double upper) {
        for (final FeedforwardLayer layer : this.layers) {
            layer.reset(lower, upper);
        }
    }
}


