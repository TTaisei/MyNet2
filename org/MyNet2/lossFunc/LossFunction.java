package org.MyNet2.lossFunc;

import org.MyNet2.*;

/**
 * Class for loss function.
 */
public class LossFunction {
    /**
     * Constructor for this class.
     */
    public LossFunction(){
        ;
    }

    /**
     * Calcurate this loss function.
     * @param y Matrix of network's output.
     * @param t Matrix of actual data.
     * @return Diiference between y and b.
     */
    public double calc(Matrix y, Matrix t){
        return y.sub(t);
    }

    /**
     * Calcurate this loss function's differential.
     * @param y Matrix of network's output.
     * @param t Matrix of actual data.
     * @return The result of differentialting the difference between y and t.
     */
    public Matrix diff(Matrix y, Matrix t){
        return y.sub(t);
    }

    @Override
    public String toString(){
        return "LossFunction";
    }
}