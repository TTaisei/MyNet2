import org.MyNet2.*;
import org.MyNet2.af.*;

public class AFTest {
    public static void main(String[] str){
        Matrix in = new Matrix(2, 3, 5.0);
        Liner af = new Liner();
        System.out.println(af.calc(in));
        System.out.println(af.diff(in));
    }
}