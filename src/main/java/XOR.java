import activation.ActivationSigmoid;
import org.jfree.data.xy.XYSeries;

public class XOR  {
    public static double XOR_INPUT[][] = {{0.0, 0.0}, {1.0, 0.0},
            {0.0, 1.0}, {1.0, 1.0}};

    public static double XOR_IDEAL[][] = {{0.0}, {1.0}, {1.0}, {0.0}};

    public static void main(String[] args) {
        FeedforwardNetwork network = new FeedforwardNetwork();
        network.addLayer(new FeedforwardLayer(2));
        network.addLayer(new FeedforwardLayer(4));
        network.addLayer(new FeedforwardLayer(1));
        network.reset(-0.5, 0.5);


        // network.addLayer(new FeedforwardLayer(new ActivationSigmoid(), 2));


        BackPropagation backPropagation = new BackPropagation(0.2, 0.9, XOR_INPUT, XOR_IDEAL, network);

        int epoch = 0;
        double error;
        XYSeries series = new XYSeries("epochs");
        do {
            error = backPropagation.train();
            System.out
                    .println("Epoch #" + epoch + " Error:" + error);
            epoch++;
            series.add(epoch, error);
        } while ((epoch < 10000) && (error > 0.05));

        System.out.println("Neural Network Results:");
        for (int i = 0; i < XOR_IDEAL.length; i++) {
            final double[] actual = network.computeOutputs(XOR_INPUT[i]);
            System.out.println(XOR_INPUT[i][0] + "," + XOR_INPUT[i][1]
                    + ", actual=" + actual[0] + ",ideal=" + XOR_IDEAL[i][0]);
        }
        Draw draw = new Draw();
        draw.draw(series);
    }




}
