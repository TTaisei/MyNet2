import org.myNet2.*;
import org.myNet2.layer.*;
import org.myNet2.actFunc.*;

public class DenseTest {
    public static void main(String[] str){
        Dense layer = new Dense(5, 7, AFType.RELU);
        System.out.println(layer);
        for (int i = 0; i < layer.w.row; i++){
            for (int j = 0; j < layer.w.col; j++){
                layer.w.matrix[i][j] = i * 0.2 - j * 0.1;
            }
        }
        System.out.println(layer.w);

        Matrix in = new Matrix(10, 5);
        for (int i = 0; i < in.row; i++){
            for (int j = 0; j < in.col; j++){
                in.matrix[i][j] = i * 0.1 - j * 0.2;
            }
        }
        System.out.println(in);
        System.out.println(layer.forward(in));
    }
}
