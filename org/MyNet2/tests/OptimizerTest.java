import java.util.Random;
import org.MyNet2.layer.*;
import org.MyNet2.actFunc.*;
import org.MyNet2.network.*;
import org.MyNet2.optimizer.*;
import org.MyNet2.lossFunc.*;
import org.MyNet2.*;

public class OptimizerTest {
    public static void main(String[] str){
        // // Dense only
        // Matrix x = new Matrix(10, 2);
        // for (int i = 0; i < x.row; i++){
        //     for (int j = 0; j < x.col; j++){
        //         x.matrix[i][j] = i*0.1 + j*0.05;
        //     }
        // }
        // Matrix t = new Matrix(10, 1);
        // for (int i = 0; i < t.row; i++){
        //     t.matrix[i][0] = x.matrix[i][0] + x.matrix[i][1];
        // }

        // Network net = new Network(
        //     2,
        //     new Dense(6, AFType.RELU),
        //     new Dense(1, AFType.LINEAR)
        // );
        // System.out.println(net);
        // GD opt = new GD(net, new MSE());
        
        // opt.fit(x, t, 5);

        // System.out.println(t);
        // System.out.println(net.forward(x));

        Matrix4d x = new Matrix4d(new int[]{10, 2, 4, 4}, new Random(0));
        Matrix t = new Matrix(10, 1, new Random());

        Network net = new Network(
            2, 4, 4,
            new Conv(4, new int[]{3, 3}, AFType.RELU),
            new MaxPooling(2),
            new Dense(4, AFType.RELU),
            new Dense(1, AFType.RELU)
        );
        net.summary();
        Matrix y = net.forward(x.flatten());

        GD opt = new GD(net, new MSE());
        opt.back(x.flatten(), y, t);
    }
}