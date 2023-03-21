import org.myNet2.*;
import org.myNet2.actFunc.*;

public class AFTest {
    public static void main(String[] str){
        Matrix in = new Matrix(2, 3, 5.0);
        Tanh af = new Tanh();
        System.out.println(af.calc(in));
        System.out.println(af.diff(in));
    }
}
