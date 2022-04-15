import java.util.Random;
import org.MyNet2.network.*;
import org.MyNet2.layer.*;
import org.MyNet2.actFunc.*;
import org.MyNet2.*;

public class NetworkTest {
    public static void main(String[] str){
        Matrix4d in = new Matrix4d(new int[]{10, 2, 4, 4}, new Random(0));

        Network net = new Network(
            2, 4, 4,
            new Conv(4, new int[]{3, 3}, AFType.RELU),
            new MaxPooling(2),
            new Dense(4, AFType.RELU),
            new Dense(1, AFType.RELU)
        );
        net.summary();
        System.out.println(net.forward(in.flatten()));
        net.save("NetworkTest.net");

        Network netLoaded = Network.load("NetworkTest.net");
        netLoaded.summary();
        System.out.println(netLoaded.forward(in.flatten()));
    }
}