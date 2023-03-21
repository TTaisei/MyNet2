import org.myNet2.layer.*;
import org.myNet2.actFunc.*;

public class Test {
    public static void main(String[] str){
        String name = new Dense(10, AFType.RELU).getClass().getName();
        System.out.println(name);
    }
}
