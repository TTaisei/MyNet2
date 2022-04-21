import org.MyNet2.layer.*;
import org.MyNet2.actFunc.*;
import org.MyNet2.network.*;
import org.MyNet2.optimizer.*;
import org.MyNet2.lossFunc.*;
import org.MyNet2.*;

public class OptimizerTest {
    public static void main(String[] str){
        Matrix x = new Matrix(10, 3);
        for (int i = 0; i < x.row; i++){
            for (int j = 0; j < x.col; j++){
                x.matrix[i][j] = i*0.1 - j*0.1;
            }
        }
        // System.out.println(x);
        Matrix t = new Matrix(10, 1);
        for (int i = 0; i < t.row; i++){
            t.matrix[i][0] = i * 0.05;
        }

        Network net = new Network(
            3,
            new Dense(5, AFType.RELU),
            new Dense(1, AFType.LINEAR)
        );
        System.out.println(net);
        Optimizer opt = new Optimizer();
        opt.net = net;
        opt.layersLength = 2;
        opt.lossFunc = new MSE();
        
        Matrix y = net.forward(x);
        // System.out.println(y);
        // System.out.println(t);
        opt.backLastLayer(x, y, t);
    }
}