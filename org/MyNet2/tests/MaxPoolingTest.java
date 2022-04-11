import org.MyNet2.*;
import org.MyNet2.layer.*;

public class MaxPoolingTest {
    public static void main(String[] str){
        int[] shape = {2, 3, 6, 6};
        Matrix4d in = new Matrix4d(shape);
        Matrix m = new Matrix(6, 6);

        for (int i = 0; i < m.row; i++){
            for (int j = 0; j < m.col; j++){
                m.matrix[i][j] = i * 0.1 - j * 0.2;
            }
        }

        for (int i = 0; i < in.shape[0]; i++){
            for (int j = 0; j < in.shape[1]; j++){
                in.matrix.get(i).matrix.set(j, m.add(i * 0.1 - j * 0.2));
            }
        }

        System.out.println(in);
        System.out.println();

        int[] returnSize = {2, 2, 3, 3};
        MaxPooling pool = new MaxPooling(1, 6, 6, 2, 2, returnSize);
        System.out.println(pool.forward(in));
    }
}