import org.MyNet2.layer.*;
import org.MyNet2.actFunc.*;
import org.MyNet2.network.*;
import org.MyNet2.optimizer.*;
import org.MyNet2.lossFunc.*;
import org.MyNet2.*;

public class OptimizerTest {
    public static void main(String[] str){
        Matrix x = new Matrix(5, 3);
        for (int i = 0; i < x.row; i++){
            for (int j = 0; j < x.col; j++){
                x.matrix[i][j] = i*0.1 - j*0.1;
            }
        }
        Matrix t = new Matrix(5, 2);
        for (int i = 0; i < t.row; i++){
            for (int j = 0; j < t.col; j++){
                t.matrix[i][j] = i*0.11 - j*0.09;
            }
        }

        Network net = new Network(
            3,
            new Dense(3, AFType.RELU),
            new Dense(2, AFType.LINEAR)
        );
        System.out.println(net);
        GD opt = new GD(net, new MSE());
        
        Matrix y = net.forward(x);
        opt.back(x, y, t);
    }
}