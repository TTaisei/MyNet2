import javax.management.relation.RelationSupport;

import org.MyNet2.*;
import org.MyNet2.layer.*;
import org.MyNet2.actFunc.*;

public class ConvTest {
    public static void main(String[] str){
        Matrix4d in = new Matrix4d(new int[]{2, 3, 6, 6});
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

        Conv conv = new Conv(3, 4, new int[]{6, 6}, new int[]{2, 2}, AFType.RELU);
        Matrix4d w = new Matrix4d(new int[]{4, 3, 2, 2});
        for (int i = 0; i < conv.kernelNum; i++){
            for (int j = 0; j < conv.channelNum; j++){
                w.matrix.get(i).matrix.set(j, new Matrix(2, 2, i * 0.1 - j * 0.2));
                System.out.println(w.matrix.get(i).matrix.get(j));
            }
            System.out.println();
        }
        conv.w = w.flatten();
        Matrix out = conv.forward(in.flatten());
        System.out.println(out.toMatrix4d(2, 4, 5, 5));
    }
}