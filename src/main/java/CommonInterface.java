import java.io.File;
import java.io.IOException;

public interface CommonInterface {
    public double outputFor(double[] x);

    public double train(double[] x, double argValue);

    public void save(File argFile);

    public void load(String argFileName) throws IOException;
}
